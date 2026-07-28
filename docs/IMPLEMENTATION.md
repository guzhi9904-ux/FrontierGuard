# Implementation notes

## Precision semantics

The research backend stores ordinary floating-point weights and applies
quantize-dequantize operators during a forward pass. This is intentional:

- BF16, W4 and W8 interventions share exactly one model object;
- no BF16 KV cache is passed into a quantized condition;
- module restoration is reversible and testable;
- fake-quant latency and memory are never reported as deployment speedups.

`PrecisionAction` controls weight, activation and KV bit widths. Linear modules
are wrapped by `FakeQuantLinear`; the original `nn.Linear` remains the only
parameter owner. A `QuantizationController` can:

- disable all quantization for a BF16 reference forward;
- assign a static `PrecisionMap`;
- temporarily restore named modules to BF16;
- temporarily switch named modules to W8.

KV fake quantization supports legacy tuple caches and Transformers
`DynamicCache`-like objects. Version-specific packed KV kernels belong in an
external backend.

## Token alignment

Teacher forcing uses:

```text
logits[:, j] -> input_ids[:, j + 1]
```

Step spans returned by `trace_input_ids` are positions in the target vector
`input_ids[:, 1:]`. Attribution converts them back to input-token indices by
adding one. Tests should be added for every new tokenizer family because a
one-token shift invalidates first-error localization.

## Intervention hierarchy

The included attribution script measures local NLL rescue. The intended full
experiment has three levels:

1. local-window rescue for mechanism;
2. post-frontier free-generation rescue for cascade prevention;
3. static precision-map free generation for paper results.

Only level 3 supports the main method claim. Local NLL rescue is used to reduce
the number of expensive rollouts.

## External backends

Adapters for QuaRot and FlatQuant implement the `QuantBackend` protocol:

```python
class QuantBackend(Protocol):
    def prepare(self, model, calibration_data=None): ...
    def set_precision_map(self, precision_map): ...
    def intervention(self, module_names, action=None, *, disable_quantization=False): ...
    def cost(self, precision_map, sequence_length): ...
    def metadata(self): ...
```

Do not import an upstream repository throughout the frontier code. Keep all
version-specific imports in one backend adapter and record the exact commit in
`third_party/locked_revisions.yaml`.

The repository additionally contains a calibrated SmoothQuant-style reference
backend. It gathers per-input-channel activation maxima on disjoint QCal data
and evaluates `(X / s)(W * s)^T` before fake quantization. It is a transparent
mechanism backend, not a replacement for the formal FlatQuant/QuaRot runs.

## 4090 execution

Recommended first run:

```bash
python scripts/00_audit_env.py
python -m frontierguard smoke-quant
python scripts/01_prepare_data.py \
  --input data/raw/profile_candidates.jsonl \
  --output artifacts/manifests/pilot.json \
  --output-dir data/splits/pilot \
  --qcal 32 --profile 20 --validation 20
python scripts/02_generate_traces.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --input data/raw/profile_candidates.jsonl \
  --output artifacts/traces/profile.jsonl \
  --max-new-tokens 8192 --seeds 0 1
python scripts/02b_calibrate_smoothquant.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --input artifacts/traces/qcal.jsonl \
  --output artifacts/calibration/sq_1p5b.safetensors \
  --alpha 0.5 --max-samples 32
python scripts/03a_precision_sweep.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --traces artifacts/traces/profile.jsonl \
  --output artifacts/frontiers/precision_ladder.jsonl \
  --backend smoothquant \
  --calibration-scales artifacts/calibration/sq_1p5b.safetensors \
  --limit 3
python scripts/03c_operating_point_sweep.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --traces artifacts/traces/profile.jsonl \
  --output artifacts/evaluation/operating_points.jsonl \
  --backend smoothquant \
  --calibration-scales artifacts/calibration/sq_1p5b.safetensors \
  --limit 5 --seeds 0
python scripts/03_scan_frontiers.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --traces artifacts/traces/profile.jsonl \
  --output artifacts/frontiers/teacher_forced.jsonl
python scripts/03b_counterfactual_frontiers.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --traces artifacts/traces/profile.jsonl \
  --output artifacts/frontiers/final.jsonl \
  --seeds 0 1 2 3 \
  --min-trustworthy-seeds 4
```

Counterfactual rollout orchestration is implemented in
`frontierguard.workflows`.
The command displays separate teacher-forcing, total-rollout and current-token
progress bars. Each completed trace is checkpointed to the output JSONL, so a
later interruption does not discard earlier traces. During cached decoding,
only the newly appended KV suffix is fake-quantized; historical entries are
not repeatedly quantized.

Legacy trace JSON is resegmented at scan time. Steps before `</think>` are
classified as reasoning; the closing tag and all later answer-presentation
content remain reconstructable but are excluded from the frontier shortlist.
Each prefix outcome stores raw per-seed continuations and a paired
difference-in-differences bootstrap interval. Runs with fewer than
`--min-trustworthy-seeds` remain valid smoke tests but cannot emit a
trustworthy frontier.

Version 0.3.1 evaluates every eligible step for short traces
(`--exhaustive-step-threshold 16`). Longer traces screen with JSD, margin and
NLL, then add neighboring steps. This supersedes the older fixed-size
shortlist recommendation above.

Answer extraction records boxed, GSM-style, explicit-final and numeric
fallback candidates. Markdown prose such as `Billy helps **240 people**` is
normalized to `240`. Every rollout records the selected extraction method,
all candidates, failure type, repetition fraction and whether EOS was
reached. Use the offline re-judge command after parser changes:

```bash
python scripts/03d_rejudge_counterfactual.py \
  --input artifacts/frontiers/old.jsonl \
  --traces artifacts/traces/profile.jsonl \
  --output artifacts/frontiers/rejudged.jsonl
```

The default teacher-forced scan evaluates one step window at a time. It keeps
exact target NLL and margin but computes JSD on an FP Top-32 plus tail
partition. This avoids retaining two full `sequence x vocabulary` tensors.
Use `scan_trace(..., low_memory=False)` only for short-sequence validation.

Module attribution consumes `recovery_frontier_step` by default and can target
the full window with `--target-scope window`. A weight rescue from W4A16 is
W8A16, not W8A8: untouched components always preserve their baseline
precision. Select the raised component explicitly with
`--rescue-component weight|activation|weight_activation`.

Attribution is candidate generation rather than the paper endpoint.
`04b_evaluate_selective_rescue.py` tests candidate precision maps by generating
from the original prompt. It pairs every condition on the same problem/seed,
uses strict EOS-aware correctness, compares against deterministic random maps
with the same layer budget, and reports

```text
(candidate accuracy - uniform-low accuracy)
------------------------------------------------
(uniform-high accuracy - uniform-low accuracy)
```

as the recovery ratio. Missing outcome rescue in local attribution is stored
as JSON `null`, not a misleading zero. Local attribution additionally stores
baseline and rescued NLL separately. Use `layer_family` grouping plus
`--include-group-regex` to split shortlisted layers before projection-level
scans.

## Known v0.1 limitations

- Dynamic per-layer KV precision is not yet wired into generation; the
  reference runner applies one default KV action to the cache.
- Teacher-forced low-memory scanning isolates W/A effects (`use_cache=False`);
  W/A/KV effects are jointly tested by counterfactual and final generation.
- The included allocator CLI consumes precomputed module utility. The
  `measured_greedy` library API supports non-additive remeasurement and should
  be used by the full experiment driver.
- Real W4/W8 storage and throughput need packed kernels.
- Intermediate semantic-step verification remains dataset-specific; final
  answer verification is conservative and implemented.
- The reference generation loop is batch-1 and prioritizes transparent cache
  handling over vLLM throughput.
