# FrontierGuard

FrontierGuard is a research framework for **first-error-frontier mixed
precision** in reasoning language models. It measures where low-bit
quantization first becomes outcome-critical, restores selected modules inside
an otherwise quantized model, and solves a budgeted static precision map.

The repository intentionally separates:

- a kernel-independent research backend for causal experiments;
- adapters for strong PTQ implementations such as QuaRot and FlatQuant;
- deployment claims, which require packed low-bit kernels and are not inferred
  from fake quantization latency.

## Research loop

```text
verified BF16 traces
  -> teacher-forced divergence scan
  -> counterfactual prefix rollouts
  -> first-error frontier
  -> module precision restoration
  -> measured budget allocation
  -> free-generation evaluation
```

## Status

Version 0.4.0 implements:

- symmetric/asymmetric groupwise fake quantization;
- calibrated SmoothQuant-style per-linear activation balancing;
- weight and activation fake-quantized `torch.nn.Linear` wrappers;
- legacy and dynamic KV-cache fake quantization utilities;
- incremental KV-cache fake quantization during autoregressive decoding;
- phase-aware reasoning/presentation segmentation and structural-step filtering;
- raw per-seed counterfactual continuations and paired bootstrap intervals;
- Markdown/LaTeX-aware answer extraction with auditable candidate provenance;
- offline re-judging of saved continuations without another model run;
- exhaustive short-CoT prefix evaluation and multi-signal long-CoT candidates;
- separate first-error, recovery-frontier and frontier-window diagnostics;
- rollout failure taxonomy for wrong answers, repetition and truncation;
- precision-map-controlled module interventions;
- frontier-conditioned BF16-to-quantized projection damage attribution;
- optional exact activation patches for predicted and matched-random modules;
- problem-level bootstrap stability and normalized-depth transfer summaries;
- BF16-correct/quantized-failing cohort construction;
- reasoning-step segmentation and trace schemas;
- JSD, margin, NLL, bypass-gain and frontier detection;
- module rescue aggregation and pairwise interaction metrics;
- exact-cost greedy mixed-precision allocation;
- paired, problem-level bootstrap confidence intervals;
- a Transformers batch-1 runner for teacher forcing and sampling;
- CLI utilities and unit/integration tests.

Real QuaRot/FlatQuant kernels are external backends. Their revisions must be
locked before publication; no third-party source is copied into this repo.
Uniform RTN is a diagnostic lower bound, not the formal W4A4 baseline. See
[the backend policy](docs/QUANTIZATION_BACKENDS.md).

## Install

```bash
conda create -n frontierguard python=3.11 -y
conda activate frontierguard
pip install -e ".[research,test]"
pytest
```

Run the CPU smoke test:

```bash
frontierguard smoke-quant
frontierguard audit-env --output environment.json
```

Before an expensive counterfactual run, execute the precision ladder:

```bash
python scripts/03a_precision_sweep.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --traces artifacts/traces/pilot.jsonl \
  --output artifacts/frontiers/precision_ladder.jsonl \
  --limit 1
```

Create deterministic, disjoint pilot splits and materialize runnable JSONL
files:

```bash
python scripts/01_prepare_data.py \
  --input data/raw/gsm8k_candidates.jsonl \
  --output artifacts/manifests/gsm8k_pilot.json \
  --output-dir data/splits/gsm8k_pilot \
  --qcal 32 --profile 20 --validation 20
```

Screen prompt-only operating points before launching prefix rollouts:

```bash
python scripts/03c_operating_point_sweep.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --traces artifacts/traces/profile.jsonl \
  --output artifacts/evaluation/operating_points.jsonl \
  --backend smoothquant \
  --calibration-scales artifacts/calibration/sq_1p5b.safetensors \
  --limit 5 --seeds 0
```

Counterfactual runs show model setup, teacher-forcing window, rollout and
per-token progress with elapsed time and ETA. Completed traces are
checkpointed to the requested JSONL output. Pass `--no-progress` only for
non-interactive jobs that do not need progress bars.

Traces with at most 16 eligible reasoning steps evaluate every adjacent
prefix by default. Longer traces use JSD, margin and NLL candidates plus their
neighbors. Output filenames that contain a `wXaYkvZ` label are checked against
the actual action to prevent mislabeled experiments.

Saved v0.3.0 counterfactual outputs can be re-judged after parser updates
without loading the model:

```bash
python scripts/03d_rejudge_counterfactual.py \
  --input artifacts/frontiers/w4a8_v030.jsonl \
  --traces artifacts/traces/profile.jsonl \
  --output artifacts/frontiers/w4a8_v031_rejudged.jsonl
```

Version 0.3.2 closes the pilot causal loop with prompt-only selective-rescue
generation. Candidate layers are raised from W4 to W8 while A8 and KV16 remain
fixed, and deterministic equal-layer-budget random maps provide the required
control:

```bash
python scripts/04b_evaluate_selective_rescue.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --traces artifacts/traces/gsm8k_pilot1.jsonl \
  --output artifacts/evaluation/pilot1_selective_rescue_v032.jsonl \
  --backend smoothquant \
  --calibration-scales artifacts/calibration/sq_pilot.safetensors \
  --rescue layer13=13 \
  --rescue layer16=16 \
  --rescue top2=13,16 \
  --rescue early2=1,2 \
  --random-budget 2 --random-maps 5 --random-seed 2026 \
  --seeds 0 1 2 3 4 5 6 7
```

The summary reports strict EOS-aware answer accuracy, paired lift over uniform
W4A8, recovery relative to uniform W8A8, exact selected parameter fraction and
the mean of matched random controls. Runs checkpoint after every generation
and resume only when the full configuration fingerprint matches.

For component dissection, attribution supports
`--grouping layer_family --include-group-regex`:

```bash
python scripts/04_run_attribution.py \
  ... \
  --grouping layer_family \
  --include-group-regex '^layer_(13|16)\.'
```

The next-stage attribution does not assume that a layer found on one trace is
globally important. Build a multi-problem failure cohort, score projection
damage at each detected frontier, and compare discovery/validation or
different architectures:

```bash
python scripts/03e_build_failure_cohort.py \
  --operating-points artifacts/evaluation/profile_operating_points.jsonl \
  --traces artifacts/traces/profile.jsonl \
  --quant-condition w4a8kv16 \
  --output-traces artifacts/traces/profile_w4a8_failure_cohort.jsonl

python scripts/04c_frontier_damage_patching.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --traces artifacts/traces/profile_w4a8_failure_cohort.jsonl \
  --frontiers artifacts/frontiers/profile_w4a8_frontiers.jsonl \
  --output artifacts/attribution/profile_w4a8kv16_damage_v040.jsonl \
  --backend smoothquant \
  --calibration-scales artifacts/calibration/sq_1p5b.safetensors \
  --weight-bits 4 --activation-bits 8 --kv-bits 16 \
  --exact-top-k 5 --exact-random-k 5

python scripts/04d_summarize_damage_stability.py \
  --scores profile=artifacts/attribution/profile_damage.jsonl \
  --scores validation=artifacts/attribution/validation_damage.jsonl \
  --output reports/damage_stability_v040.json
```

The gradient score is explicitly labeled as an STE first-order estimator.
Exact patches and final prompt-only selective-rescue generation remain
separate validation stages.

Version 0.4.1 connects those stages without manually guessing layers.
`04b_evaluate_selective_rescue.py --damage-scores` averages damage within each
problem, filters projections by problem coverage and positive-sign
consistency, and builds cumulative Top-K W8 weight exceptions on the W4A8KV16
backbone. Each Top-K map is compared with deterministic random maps matched on
projection type and module count:

```bash
python scripts/04b_evaluate_selective_rescue.py \
  --model "$MODEL_DIR" \
  --traces artifacts/traces/gsm8k_clean_fail15_seed0_v040.jsonl \
  --damage-scores artifacts/attribution/gsm8k_fail15_w4a8kv16_damage_v040.jsonl \
  --output artifacts/evaluation/gsm8k_fail15_ranked_rescue_v041.jsonl \
  --backend smoothquant \
  --calibration-scales artifacts/calibration/sq_gsm8k_qcal32_v040.safetensors \
  --weight-bits 4 --activation-bits 8 --kv-bits 16 \
  --high-weight-bits 8 \
  --rank-budgets 2 4 8 \
  --ranked-random-maps 2 \
  --minimum-module-problem-fraction 0.8 \
  --minimum-module-positive-fraction 0.5 \
  --random-maps 0 \
  --seeds 0 1 2 3 \
  --max-new-tokens 2048
```

The output explicitly labels selection/evaluation overlap as
`in_sample_exploratory`. Re-run the frozen ranked map on new problems to obtain
a held-out result; local NLL ranking alone is never reported as final task
recovery.

To test activation balancing, calibrate on a disjoint QCal split and select the
calibrated backend:

```bash
python scripts/02b_calibrate_smoothquant.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --input artifacts/traces/qcal.jsonl \
  --output artifacts/calibration/sq_1p5b.safetensors
python scripts/03_scan_frontiers.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --traces artifacts/traces/profile.jsonl \
  --output artifacts/frontiers/sq_scan.jsonl \
  --backend smoothquant \
  --calibration-scales artifacts/calibration/sq_1p5b.safetensors
```

Inspect a configuration:

```bash
frontierguard show-config configs/experiment/e2_frontier.yaml
```

## Initial experiment matrix

Models:

1. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` for pipeline validation and
   model-size sensitivity, not the primary claim.
2. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` for primary experiments.
3. `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` for architecture transfer.

Evaluation:

- GSM8K for sanity checks;
- MATH-500 as the primary benchmark;
- AIME-120 as the difficult reasoning benchmark;
- AIME-2026 only after the method is frozen.

Quantization:

- BF16;
- W8A8KV8;
- RTN/GPTQ/AWQ W4A16;
- uniform, QuaRot and FlatQuant W4A4KV4;
- FrontierGuard W4 backbone with selected W8 modules.

See [docs/EXPERIMENT_SPEC.md](docs/EXPERIMENT_SPEC.md) and
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## Reproducibility rules

- Quantizer calibration, frontier profiling and validation splits are disjoint.
- MATH-500 and AIME are never used for module selection.
- A quantized run never consumes a BF16 KV cache.
- Local-window rescue is mechanism evidence; paper accuracy comes from static
  precision maps and free generation from the prompt.
- Weight, activation and KV cost are reported separately.
- Bootstrap resamples problems, not individual generations from the same
  problem.

## License

Apache-2.0.
