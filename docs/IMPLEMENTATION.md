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

Step spans returned by `trace_input_ids` are logit positions in the
teacher-forcing target vector: logit position `j` predicts input token
`j + 1`. The legacy local-NLL helper accepts input-token indices and therefore
adds one; compression-damage patching consumes the logit spans directly.
Tests should be added for every new tokenizer family because a one-token shift
invalidates first-error localization.

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

### Multi-problem compression-damage experiment

Do not infer stable modules from a single pilot trace. First screen a prompt-
only operating point, then materialize the BF16-correct/quantized-failing
cohort:

```bash
python scripts/03e_build_failure_cohort.py \
  --operating-points artifacts/evaluation/profile_operating_points.jsonl \
  --traces artifacts/traces/profile.jsonl \
  --quant-condition w4a8kv16 \
  --output-traces artifacts/traces/profile_w4a8_failure_cohort.jsonl \
  --minimum-problems 20
```

Run counterfactual frontier detection on that cohort, then score every named
Transformer projection with one BF16 forward and one quantized surrogate
backward per trace. Only predicted top modules and matched projection controls
receive expensive exact activation patches:

```bash
python scripts/04c_frontier_damage_patching.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --traces artifacts/traces/profile_w4a8_failure_cohort.jsonl \
  --frontiers artifacts/frontiers/profile_w4a8_frontiers.jsonl \
  --output artifacts/attribution/profile_w4a8kv16_damage_v040.jsonl \
  --backend smoothquant \
  --calibration-scales artifacts/calibration/sq_1p5b.safetensors \
  --weight-bits 4 --activation-bits 8 --kv-bits 16 \
  --target-scope window --max-window-tokens 128 \
  --exact-top-k 5 --exact-random-k 5
```

The 128-token cap is centered on the selected frontier step and bounds CPU
capture and backward memory. Reduce it to 64 for 7B/8B models if necessary.
The estimator uses an identity STE through activation fake quantizers; the
JSON records that surrogate explicitly. Forward values remain ordinary
quantize/dequantize values.

Run profile and validation independently. The first `--scores` input is the
discovery split:

```bash
python scripts/04d_summarize_damage_stability.py \
  --scores profile=artifacts/attribution/profile_damage.jsonl \
  --scores validation=artifacts/attribution/validation_damage.jsonl \
  --scores llama8b=artifacts/attribution/llama8b_damage.jsonl \
  --output reports/damage_stability_v040.json \
  --top-k 10 --depth-bins 4
```

The summary averages repeated trace seeds inside each problem, bootstraps
problems, reports exact-module and normalized-depth stability separately, and
labels fewer than 20 problems as pilot-only evidence. Selected projections
must still pass `04b_evaluate_selective_rescue.py` from the original prompt;
activation patching is mechanism evidence, not final accuracy.

### Damage-ranked static rescue

Version 0.4.1 lets the prompt-only evaluator consume the `04c` JSONL directly.
For each projection it first averages repeated trace seeds inside a problem,
then ranks the problem-level means. A projection is eligible only when it
meets both `--minimum-module-problem-fraction` and
`--minimum-module-positive-fraction`; `--require-positive-module-ci` is
available for larger cohorts. Cumulative `--rank-budgets` promote only the
selected weights from W4 to W8, leaving A8 and KV16 unchanged.

Random controls are sampled from the same projection types as the selected
modules. This makes `ranked_topK` versus `ranked_topK_random_*` a comparison of
where precision is spent, rather than a comparison with a different module
count or attention/MLP mix. The summary stores exact selected module names,
parameter-weighted effective bits, selection problem IDs, and overlap with the
generation evaluation cohort.

Using the same 15 problems for ranking and generation is a valid causal-loop
pilot, but its summary is deliberately labeled `in_sample_exploratory`.
Publication evidence requires freezing the selected module map and evaluating
it on unseen problems, ideally on the 7B primary model and a second
architecture using normalized-depth/projection-family transfer.

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
