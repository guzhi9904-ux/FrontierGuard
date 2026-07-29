# Experiment specification

## Primary claim

FrontierGuard tests whether compression damage measured at a first-error
frontier predicts budget-controlled reasoning recovery more effectively than
generic sensitivity metrics. It does not assume in advance that one fixed set
of Transformer layers is responsible for every failure.

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

A trustworthy legacy detector frontier is the earliest step jointly supported by:

- teacher-forced Jensen-Shannon divergence;
- target-token margin drop;
- positive \(B_s\), with a positive confidence lower bound.

Only semantic steps inside the reasoning phase are eligible. For tagged
reasoning models, content after `</think>` is retained for final-answer
verification but classified as presentation. Pure headings, answer labels and
markup transitions are never reasoning-frontier candidates.

Version 0.3.1 reports three non-interchangeable locations:

- `first_error_step`: the earliest teacher-forced NLL-onset candidate;
- `causal_first_error_step`: the earliest positive quantization-specific
  paired prefix gain;
- `recovery_frontier_step`: the largest positive paired prefix gain.

`frontier_window` is the inclusive span covering these diagnostics. The
teacher-forced location is a screening candidate rather than a calibrated
hypothesis test. A recovery frontier is trustworthy only when its paired
confidence lower bound is positive.

For at most 16 eligible reasoning steps, every adjacent prefix is evaluated.
Longer traces use the union of JSD/margin and NLL candidates, expanded to
neighboring steps. This prevents a behaviorally important step from being
silently removed by one teacher-forced ranking signal.

More verified prefix normally improves success, so the experiment does not
define a frontier as "failure probability rising with prefix length."

## Causal intervention

Attribution starts from the complete quantized model \(Q\), then restores a
module:

\[
Q^{m\leftarrow h}.
\]

Restoring one module inside \(Q\) is the correct intervention. Quantizing one
module inside a BF16 model answers a different question because the module sees
a BF16 context.

The deployable first-stage action raises selected components from the active
low-precision operating point, for example W4 to W8 while leaving A8/KV16
unchanged. BF16 activation patching is an oracle mechanism test, not the
deployed precision action.

Version 0.4.0 adds compression-damage attribution. For projection \(m\) and
frontier-token set \(T\), it records BF16 and quantized outputs under an
identical verified prefix and estimates:

\[
\widehat R_m =
-\nabla_{h_m^Q}L_Q(T)^\top(h_m^F-h_m^Q).
\]

A positive value predicts that replacing the quantized projection output by
its BF16 counterpart lowers frontier-window NLL. Activation fake quantizers
use an identity straight-through gradient for this estimator only; exact
activation patches validate selected scores with unchanged quantized forward
values. Neither score proves that a module uniquely stores a reasoning skill.

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
- GPTQ/AWQ W4A16 weight-only baselines;
- SmoothQuant W8A8 weight-activation baseline;
- W4A8KV16 as a partially recoverable mechanism operating point, not the only
  paper setting;
- uniform RTN W4A4KV4 as a diagnostic lower bound only;
- FlatQuant W4A4 as the primary strong PTQ backend;
- QuaRot W4A4KV4 as a rotation-based transfer backend;
- W4 backbone plus FrontierGuard-selected W8 modules.

The 1.5B model is used for pipeline debugging and model-size sensitivity. The
7B model is the primary claim model; Llama-8B tests architecture transfer.
Formal attribution does not proceed when the precision ladder shows global
distribution saturation (most step JSD values at or above 0.68).
It also does not proceed from a prompt-only operating point whose quantized
accuracy is effectively zero or one; the counterfactual detector needs a
partially recoverable regime.

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
- first-order damage scores predict exact patch rescue better than matched
  random projections on held-out problems.

## Statistics

- paired seeds across conditions;
- store each paired seed transition, continuation, extracted answer and
  truncation flag rather than only aggregated success counts;
- store answer-extraction method, all answer candidates, repetition score and
  failure type for every rollout;
- require EOS for strict rollout success; a correct number inside a truncated
  continuation is retained as diagnostic `answer_correct` evidence only;
- require at least 4 paired seeds before a per-trace frontier can be labeled
  trustworthy;
- bootstrap resampling at the problem level;
- average repeated traces/seeds inside a problem before module inference;
- AIME seeds are aggregated within problem before resampling;
- report 95% confidence intervals;
- report accuracy, gap recovery, output tokens, token inflation, truncation,
  effective bits, KV bytes/token and peak memory.

`confidence_level` names the interval coverage (normally 0.95); it is not the
posterior probability that a detected step is correct. `evidence_score`
retains the detector's heuristic ranking score and must not be reported as a
calibrated probability.

## Scope guard

Fake quantization validates causal and accuracy claims only. End-to-end memory
and latency claims require a real packed mixed W4/W8 backend.

Exact layer-number transfer is not a primary claim. Cross-architecture
analysis uses normalized depth, attention/MLP family and projection type.
