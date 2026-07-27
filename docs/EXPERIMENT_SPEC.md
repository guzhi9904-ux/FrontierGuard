# Experiment specification

## Primary claim

FrontierGuard tests whether first-error-frontier causal rescue selects a small,
stable set of high-precision Transformer modules that restores reasoning
accuracy more effectively than generic sensitivity metrics at equal cost.

The publishable loop is:

```text
measure -> locate -> intervene -> allocate -> verify
```

Finding a first divergent token or a first faulty reasoning step is not, by
itself, the contribution.

## Formal frontier

For a verified BF16 trace with prefixes \(P_s\), define final-answer success:

\[
u_s^M = \Pr_M(\text{correct final answer}\mid P_s), \quad M\in\{F,Q\}.
\]

The gain from injecting verified step \(s\) is:

\[
G_s^M=u_s^M-u_{s-1}^M.
\]

The quantization-specific bypass gain is:

\[
B_s=G_s^Q-G_s^F.
\]

A trustworthy frontier is the earliest step jointly supported by:

- teacher-forced Jensen-Shannon divergence;
- target-token margin drop;
- positive \(B_s\), with a positive confidence lower bound.

More verified prefix normally improves success, so the experiment does not
define a frontier as “failure probability rising with prefix length.”

## Causal intervention

Attribution starts from the complete quantized model \(Q\), then restores a
module:

\[
Q^{m\leftarrow h}.
\]

Restoring one module inside \(Q\) is the correct intervention. Quantizing one
module inside a BF16 model answers a different question because the module sees
a BF16 context.

The deployable first-stage action is W8A8/KV8. BF16 restoration is an oracle
upper bound.

## Frozen initial matrix

Models:

- DeepSeek-R1-Distill-Qwen-1.5B: pipeline;
- DeepSeek-R1-Distill-Qwen-7B: primary;
- DeepSeek-R1-Distill-Llama-8B: architecture transfer.

Disjoint NuminaMath splits:

- QCal 128;
- FEF-Profile 96;
- FEF-Validation 32.

Evaluation:

- GSM8K sanity;
- MATH-500 primary;
- AIME-120 difficult reasoning;
- AIME-2026 optional blind test after method freeze.

Quantization:

- BF16 and W8A8KV8 controls;
- RTN/GPTQ/AWQ W4A16;
- uniform, QuaRot and FlatQuant W4A4KV4;
- W4 backbone plus FrontierGuard-selected W8 modules.

Generation starts at temperature 0.6, top-p 0.95 and 8192 maximum new tokens.
Raise the limit if truncation exceeds 2%. Greedy decoding is not used for the
DeepSeek-R1 distilled models.

## Equal-cost baselines

- random selection with 10 seeds;
- first/last layers;
- structural projection rules;
- Hessian/GPTQ sensitivity;
- activation magnitude/AWQ saliency;
- reconstruction error;
- final-distribution KL;
- small-set oracle as an upper bound only.

Report weight bits, activation peak and KV bytes/token separately.

## GO criteria

- a stable quantization gap of at least 3 percentage points;
- the frontier detector beats position and single-signal baselines;
- the top 10% modules recover at least 30% of recoverable failures;
- equal-cost FrontierGuard beats Hessian, saliency and output-KL selectors;
- local rescue transfers to static free-generation gains;
- the NuminaMath precision map transfers to MATH-500 and AIME;
- Top-10% split Jaccard is at least 0.4.

## Statistics

- paired seeds across conditions;
- bootstrap resampling at the problem level;
- AIME seeds are aggregated within problem before resampling;
- report 95% confidence intervals;
- report accuracy, gap recovery, output tokens, token inflation, truncation,
  effective bits, KV bytes/token and peak memory.

## Scope guard

Fake quantization validates causal and accuracy claims only. End-to-end memory
and latency claims require a real packed mixed W4/W8 backend.
