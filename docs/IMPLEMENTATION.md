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
python scripts/03_scan_frontiers.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --traces artifacts/traces/profile.jsonl \
  --output artifacts/frontiers/teacher_forced.jsonl
python scripts/03b_counterfactual_frontiers.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --traces artifacts/traces/profile.jsonl \
  --output artifacts/frontiers/final.jsonl \
  --seeds 0 1 2 3
```

Counterfactual rollout should first be run on the 3–5 shortlisted steps per
trace. The high-level implementation is in `frontierguard.workflows`.

The default teacher-forced scan evaluates one step window at a time. It keeps
exact target NLL and margin but computes JSD on an FP Top-32 plus tail
partition. This avoids retaining two full `sequence × vocabulary` tensors.
Use `scan_trace(..., low_memory=False)` only for short-sequence validation.

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
