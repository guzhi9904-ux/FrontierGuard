"""Frontier-conditioned compression-damage attribution and activation patching."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F

from frontierguard.models.adapters import ModuleDescriptor
from frontierguard.quant.controller import QuantizationController


@dataclass(frozen=True)
class DamageMeasurement:
    """First-order compression damage for one projection."""

    module_name: str
    layer_index: int
    relative_depth: float
    family: str
    projection: str
    parameter_count: int
    predicted_nll_rescue: float
    gradient_delta_dot: float
    gradient_l2: float
    activation_delta_l2: float
    activation_relative_l2: float
    gradient_delta_cosine: float
    exact_patch_nll: float | None = None
    exact_nll_rescue: float | None = None
    exact_patch_margin: float | None = None
    exact_margin_gain: float | None = None
    exact_role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DamageAttributionResult:
    bf16_nll: float
    quantized_nll: float
    bf16_margin: float
    quantized_margin: float
    measurements: tuple[DamageMeasurement, ...]


def _forward_window(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_start: int,
    target_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable counterpart of ``HFRunner.teacher_forcing_window``."""

    if target_start < 0 or target_end <= target_start or target_end >= input_ids.shape[-1]:
        raise ValueError("invalid teacher-forcing target window")
    keep = target_end - target_start
    prefix = input_ids[:, :target_end]
    parameters = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {"input_ids": prefix, "use_cache": False}
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = keep
    elif "num_logits_to_keep" in parameters:
        kwargs["num_logits_to_keep"] = keep
    logits = model(**kwargs).logits
    selected = logits if logits.shape[1] == keep else logits[:, target_start:target_end, :]
    targets = input_ids[:, target_start + 1 : target_end + 1]
    if selected.shape[:-1] != targets.shape:
        raise RuntimeError(
            f"window alignment mismatch: logits {selected.shape}, targets {targets.shape}"
        )
    return selected, targets


def _nll_tensor(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="mean",
    )


def _mean_correct_margin(logits: torch.Tensor, targets: torch.Tensor) -> float:
    scores = logits.float()
    target_scores = scores.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    top_values, top_indices = scores.topk(k=2, dim=-1)
    best_incorrect = torch.where(
        top_indices[..., 0].eq(targets),
        top_values[..., 1],
        top_values[..., 0],
    )
    return float((target_scores - best_incorrect).mean().item())


def _capture_full_precision(
    model: torch.nn.Module,
    controller: QuantizationController,
    input_ids: torch.Tensor,
    target_start: int,
    target_end: int,
    module_names: Sequence[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor):
            captured[name] = (
                output[:, target_start:target_end, :].detach().to("cpu", copy=True)
            )

        return capture

    for name in module_names:
        handles.append(controller.wrappers[name].register_forward_hook(hook(name)))
    try:
        with torch.inference_mode(), controller.disabled():
            logits, targets = _forward_window(
                model, input_ids, target_start, target_end
            )
            return captured, logits.detach(), targets.detach()
    finally:
        for handle in handles:
            handle.remove()


def _quantized_gradients(
    model: torch.nn.Module,
    controller: QuantizationController,
    input_ids: torch.Tensor,
    target_start: int,
    target_end: int,
    module_names: Sequence[str],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    quantized_outputs: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor):
            if not output.requires_grad:
                output.requires_grad_(True)
            quantized_outputs[name] = (
                output[:, target_start:target_end, :].detach().to("cpu", copy=True)
            )

            def save_gradient(gradient: torch.Tensor) -> None:
                gradients[name] = (
                    gradient[:, target_start:target_end, :]
                    .detach()
                    .to("cpu", copy=True)
                )

            output.register_hook(save_gradient)

        return capture

    for name in module_names:
        handles.append(controller.wrappers[name].register_forward_hook(hook(name)))
    parameter_states = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    for parameter, _enabled in parameter_states:
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad(), controller.straight_through_gradients():
            logits, targets = _forward_window(
                model, input_ids, target_start, target_end
            )
            loss = _nll_tensor(logits, targets)
            loss.backward()
            return quantized_outputs, gradients, logits.detach(), targets.detach()
    finally:
        for handle in handles:
            handle.remove()
        for parameter, enabled in parameter_states:
            parameter.requires_grad_(enabled)
        model.zero_grad(set_to_none=True)


def _exact_patch(
    model: torch.nn.Module,
    controller: QuantizationController,
    input_ids: torch.Tensor,
    target_start: int,
    target_end: int,
    module_name: str,
    full_precision_output: torch.Tensor,
) -> tuple[float, float]:
    replacement = full_precision_output

    def patch(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        patched = output.clone()
        patched[:, target_start:target_end, :] = replacement.to(
            device=output.device,
            dtype=output.dtype,
        )
        return patched

    handle = controller.wrappers[module_name].register_forward_hook(patch)
    try:
        with torch.inference_mode():
            logits, targets = _forward_window(
                model, input_ids, target_start, target_end
            )
        return float(_nll_tensor(logits, targets).item()), _mean_correct_margin(
            logits, targets
        )
    finally:
        handle.remove()


def _matched_random_modules(
    ranked: Sequence[ModuleDescriptor],
    selected: Sequence[str],
    *,
    count: int,
    generator: torch.Generator,
) -> list[str]:
    if count <= 0:
        return []
    by_name = {item.name: item for item in ranked}
    excluded = set(selected)
    controls: list[str] = []
    for selected_name in selected:
        if len(controls) >= count:
            break
        descriptor = by_name[selected_name]
        candidates = [
            item.name
            for item in ranked
            if item.name not in excluded
            and item.name not in controls
            and item.projection == descriptor.projection
        ]
        if not candidates:
            candidates = [
                item.name
                for item in ranked
                if item.name not in excluded
                and item.name not in controls
                and item.family == descriptor.family
            ]
        if not candidates:
            continue
        index = int(torch.randint(len(candidates), (1,), generator=generator).item())
        controls.append(candidates[index])
    return controls


def measure_compression_damage(
    model: torch.nn.Module,
    controller: QuantizationController,
    input_ids: torch.Tensor,
    target_start: int,
    target_end: int,
    descriptors: Sequence[ModuleDescriptor],
    *,
    exact_top_k: int = 0,
    exact_random_k: int = 0,
    random_seed: int = 0,
) -> DamageAttributionResult:
    """Measure all projection scores with one BF16 and one surrogate backward pass.

    The first-order score is ``-grad(L_q) dot (h_fp - h_q)``. Positive values
    predict that replacing the quantized activation with its BF16 counterpart
    lowers frontier-window NLL. Exact patches are optional validation and are
    never conflated with the first-order score.
    """

    if not descriptors:
        raise ValueError("compression-damage attribution needs at least one module")
    module_names = [item.name for item in descriptors]
    unknown = sorted(set(module_names) - controller.wrappers.keys())
    if unknown:
        raise KeyError(f"attribution modules are not instrumented: {unknown[:3]}")

    fp_outputs, fp_logits, fp_targets = _capture_full_precision(
        model,
        controller,
        input_ids,
        target_start,
        target_end,
        module_names,
    )
    q_outputs, gradients, q_logits, q_targets = _quantized_gradients(
        model,
        controller,
        input_ids,
        target_start,
        target_end,
        module_names,
    )
    if not torch.equal(fp_targets.cpu(), q_targets.cpu()):
        raise RuntimeError("BF16 and quantized target windows differ")
    missing = sorted(
        set(module_names) - fp_outputs.keys()
        | set(module_names) - q_outputs.keys()
        | set(module_names) - gradients.keys()
    )
    if missing:
        raise RuntimeError(f"missing activation/gradient captures: {missing[:3]}")

    maximum_layer = max(item.layer_index for item in descriptors)
    measurements: list[DamageMeasurement] = []
    for descriptor in descriptors:
        fp = fp_outputs[descriptor.name].float()
        quantized = q_outputs[descriptor.name].float()
        gradient = gradients[descriptor.name].float()
        delta = fp - quantized
        delta_l2 = float(delta.norm().item())
        gradient_l2 = float(gradient.norm().item())
        dot = float((gradient * delta).sum().item())
        denominator = max(delta_l2 * gradient_l2, 1e-12)
        measurements.append(
            DamageMeasurement(
                module_name=descriptor.name,
                layer_index=descriptor.layer_index,
                relative_depth=(
                    descriptor.layer_index / maximum_layer
                    if maximum_layer > 0
                    else 0.0
                ),
                family=descriptor.family,
                projection=descriptor.projection,
                parameter_count=descriptor.parameter_count,
                predicted_nll_rescue=-dot,
                gradient_delta_dot=dot,
                gradient_l2=gradient_l2,
                activation_delta_l2=delta_l2,
                activation_relative_l2=(
                    delta_l2 / max(float(fp.norm().item()), 1e-12)
                ),
                gradient_delta_cosine=dot / denominator,
            )
        )

    ranked = sorted(
        measurements,
        key=lambda item: (-item.predicted_nll_rescue, item.module_name),
    )
    top_names = [
        item.module_name
        for item in ranked[: max(0, exact_top_k)]
        if item.predicted_nll_rescue > 0
    ]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    random_names = _matched_random_modules(
        descriptors,
        top_names,
        count=exact_random_k,
        generator=generator,
    )
    exact_roles = {name: "predicted_top" for name in top_names}
    exact_roles.update({name: "matched_random" for name in random_names})

    q_nll = float(_nll_tensor(q_logits, q_targets).item())
    q_margin = _mean_correct_margin(q_logits, q_targets)
    enriched: list[DamageMeasurement] = []
    for measurement in measurements:
        role = exact_roles.get(measurement.module_name)
        if role is None:
            enriched.append(measurement)
            continue
        patch_nll, patch_margin = _exact_patch(
            model,
            controller,
            input_ids,
            target_start,
            target_end,
            measurement.module_name,
            fp_outputs[measurement.module_name],
        )
        enriched.append(
            DamageMeasurement(
                **{
                    **measurement.to_dict(),
                    "exact_patch_nll": patch_nll,
                    "exact_nll_rescue": q_nll - patch_nll,
                    "exact_patch_margin": patch_margin,
                    "exact_margin_gain": patch_margin - q_margin,
                    "exact_role": role,
                }
            )
        )
    return DamageAttributionResult(
        bf16_nll=float(_nll_tensor(fp_logits, fp_targets).item()),
        quantized_nll=q_nll,
        bf16_margin=_mean_correct_margin(fp_logits, fp_targets),
        quantized_margin=q_margin,
        measurements=tuple(enriched),
    )
