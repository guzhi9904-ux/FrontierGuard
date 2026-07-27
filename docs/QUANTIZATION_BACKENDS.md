# Quantization backend policy

FrontierGuard separates mechanism experiments from claims about competitive
post-training quantization (PTQ).

## Backend roles

| Backend | Setting | Role |
| --- | --- | --- |
| Uniform group-wise RTN | W4A4, W4A8, W4A16 | diagnostic lower bound and stress test |
| SmoothQuant-style reference | W8A8, exploratory W4A4 | calibrated fake-quant mechanism backend |
| GPTQ or AWQ | W4A16 | formal weight-only baseline |
| SmoothQuant | W8A8 | formal 8-bit weight-activation baseline |
| FlatQuant | W4A4 | preferred formal weight-activation baseline |
| QuaRot | W4A4KV4 | rotation-based transfer baseline |

GPTQ and AWQ optimize weight quantization. Combining either method with naive
A4 does not make it a competitive W4A4 method because activation outliers
remain untreated.

The in-repository SmoothQuant-style implementation applies a per-linear
equivalent transformation:

```text
Y = X W^T = (X / s) (W * s)^T
```

The channel scale `s` is calibrated from disjoint QCal traces. This backend is
useful for testing whether activation balancing removes RTN saturation, but it
must not be presented as FlatQuant or as a state-of-the-art packed INT4
implementation.

## Required experiment labels

Every scan, counterfactual run, attribution observation and generation record
stores backend metadata. Paper tables must distinguish:

- `rtn_reference_fake`;
- `smoothquant_reference_fake`;
- exact external backend name and locked revision;
- packed-kernel deployment runs.

Fake-quant results support causal and accuracy conclusions only. They do not
support latency, memory-footprint or integer-throughput claims.

## Saturation gate

The first-error detector is not meaningful when all steps already have nearly
maximal Jensen-Shannon divergence. Run `03a_precision_sweep.py` first. If most
steps exceed JSD 0.68 under natural logarithms, the backend is treated as
globally collapsed and counterfactual attribution is paused.

## Strong external backends

FlatQuant and QuaRot remain external because their architecture transforms and
packed kernels must match an exact upstream revision. Freeze revisions in
`third_party/locked_revisions.yaml` before formal runs. Do not silently emulate
either method with uniform RTN.

Primary references:

- GPTQ: <https://arxiv.org/abs/2210.17323>
- SmoothQuant: <https://arxiv.org/abs/2211.10438>
- QuaRot: <https://arxiv.org/abs/2404.00456>
- FlatQuant: <https://arxiv.org/abs/2410.09426>
- Quantized reasoning models: <https://arxiv.org/abs/2504.04823>
