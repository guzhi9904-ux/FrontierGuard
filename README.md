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

Version 0.1 implements:

- symmetric/asymmetric groupwise fake quantization;
- weight and activation fake-quantized `torch.nn.Linear` wrappers;
- legacy and dynamic KV-cache fake quantization utilities;
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

Inspect a configuration:

```bash
frontierguard show-config configs/experiment/e2_frontier.yaml
```

## Initial experiment matrix

Models:

1. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` for pipeline validation.
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
