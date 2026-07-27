# Third-party backends

FrontierGuard does not copy QuaRot, FlatQuant or Quantized-Reasoning-Models
sources. Clone them separately at the revisions recorded in
`locked_revisions.yaml`, then implement or enable the corresponding adapter.

This separation keeps causal research code independent from deployment kernels
and preserves upstream licenses.
