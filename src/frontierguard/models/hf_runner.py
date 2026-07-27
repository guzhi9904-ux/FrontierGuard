"""Batch-1 Transformers runner used by the research backend."""

from __future__ import annotations

import contextlib
import inspect
import time
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from frontierguard.quant.controller import QuantizationController
from frontierguard.quant.kv_cache import fake_quantize_kv_cache


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 8192
    seed: int = 0


def sample_top_p(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    scores = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    remove = cumulative - probabilities > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered_probabilities = torch.softmax(sorted_logits, dim=-1)
    sampled_position = torch.multinomial(
        filtered_probabilities, num_samples=1, generator=generator
    )
    return sorted_indices.gather(-1, sampled_position)


class HFRunner:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        controller: QuantizationController | None = None,
        kv_bits: int = 16,
        kv_group_size: int = 128,
        kv_symmetric: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.controller = controller
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.kv_symmetric = kv_symmetric

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        dtype: str = "bfloat16",
        device_map: str | dict[str, Any] | None = "auto",
        trust_remote_code: bool = False,
    ) -> "HFRunner":
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        dtype_argument = (
            {"dtype": torch_dtype}
            if Version(transformers.__version__) >= Version("4.56.0")
            else {"torch_dtype": torch_dtype}
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **dtype_argument,
        )
        model.eval()
        return cls(model, tokenizer)

    @contextlib.contextmanager
    def full_precision(self) -> Iterator[None]:
        previous_kv_bits = self.kv_bits
        self.kv_bits = 16
        try:
            if self.controller is None:
                yield
            else:
                with self.controller.disabled():
                    yield
        finally:
            self.kv_bits = previous_kv_bits

    def encode_chat(self, problem: str, *, enable_thinking: bool = True) -> torch.Tensor:
        messages = [{"role": "user", "content": problem}]
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking, **kwargs
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        return encoded.to(self.device)

    @torch.inference_mode()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids.to(self.device), use_cache=False).logits

    @torch.inference_mode()
    def teacher_forcing(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = input_ids.to(self.device)
        logits = self.model(input_ids=ids, use_cache=False).logits
        if ids.shape[-1] < 2:
            raise ValueError("teacher forcing needs at least two tokens")
        return logits[:, :-1, :], ids[:, 1:]

    @torch.inference_mode()
    def teacher_forcing_window(
        self,
        input_ids: torch.Tensor,
        target_start: int,
        target_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return only logits needed for one target-vector window.

        When the model exposes `logits_to_keep` or `num_logits_to_keep`, the LM
        head does not materialize prefix logits. Otherwise the method falls
        back to slicing a normal forward output.
        """

        ids = input_ids.to(self.device)
        if target_start < 0 or target_end <= target_start or target_end >= ids.shape[-1]:
            raise ValueError("invalid teacher-forcing target window")
        keep = target_end - target_start
        prefix = ids[:, :target_end]
        parameters = inspect.signature(self.model.forward).parameters
        kwargs: dict[str, Any] = {"input_ids": prefix, "use_cache": False}
        if "logits_to_keep" in parameters:
            kwargs["logits_to_keep"] = keep
        elif "num_logits_to_keep" in parameters:
            kwargs["num_logits_to_keep"] = keep
        logits = self.model(**kwargs).logits
        if logits.shape[1] == keep:
            selected = logits
        else:
            selected = logits[:, target_start:target_end, :]
        targets = ids[:, target_start + 1 : target_end + 1]
        if selected.shape[:-1] != targets.shape:
            raise RuntimeError(
                f"window alignment mismatch: logits {selected.shape}, targets {targets.shape}"
            )
        return selected, targets

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling: SamplingConfig,
    ) -> dict[str, Any]:
        if prompt_ids.shape[0] != 1:
            raise ValueError("reference runner currently supports batch size 1")
        device = self.device
        generator = torch.Generator(device=device)
        generator.manual_seed(sampling.seed)
        generated: list[int] = []
        current = prompt_ids.to(device)
        cache = None
        eos_ids = self.tokenizer.eos_token_id
        eos_set = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        started = time.perf_counter()

        for _ in range(sampling.max_new_tokens):
            output = self.model(input_ids=current, past_key_values=cache, use_cache=True)
            logits = output.logits[:, -1, :]
            cache = fake_quantize_kv_cache(
                output.past_key_values,
                self.kv_bits,
                group_size=self.kv_group_size,
                symmetric=self.kv_symmetric,
            )
            next_token = sample_top_p(
                logits,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                generator=generator,
            )
            token = int(next_token.item())
            generated.append(token)
            current = next_token
            if token in eos_set:
                break

        elapsed = time.perf_counter() - started
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "text": text,
            "token_ids": generated,
            "output_tokens": len(generated),
            "truncated": len(generated) == sampling.max_new_tokens,
            "latency_seconds": elapsed,
        }
