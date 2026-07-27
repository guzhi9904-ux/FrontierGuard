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

Version 0.3.0 implements:

- symmetric/asymmetric groupwise fake quantization;
- calibrated SmoothQuant-style per-linear activation balancing;
- weight and activation fake-quantized `torch.nn.Linear` wrappers;
- legacy and dynamic KV-cache fake quantization utilities;
- incremental KV-cache fake quantization during autoregressive decoding;
- phase-aware reasoning/presentation segmentation and structural-step filtering;
- raw per-seed counterfactual continuations and paired bootstrap intervals;
- precision-map-controlled module interventions;
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
