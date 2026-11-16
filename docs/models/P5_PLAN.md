# P5 plan: local-global temporal Transformer

Date: 2026-11-16

## Outcome and current status

P5 is the **Transformer approach** named in the Sportec model plan. It is not
GitHub PR 5, which was the dead-ball input-column change.

P5 was not complete when this plan was written. The repository had no P5 model,
training command, test, result document, branch, or pull request. The shared
prerequisites were present on `main`: the fixed Sportec split, common input
contract, common Viterbi decoder, common scorer, and keep rule. P3, P4, and P6
were still separate open pull requests, so P5 must not silently depend on their
unmerged code.

## Question P5 must answer

> With the same leak-safe flat inputs, training data, split, missingness
> treatment, temporal context, three-seed ensemble, scorer, and smoother as P4,
> does a low-capacity hybrid of local temporal convolution and global
> self-attention recover possession spells more reliably than the bidirectional
> GRU without learning match/tracker identity?

This is an architecture comparison, not a mandate to ship a Transformer. A
negative, reproducible result completes P5 if the stop rule is applied and the
evidence is recorded.

## Evidence and constraints

### Repository evidence

- `docs/common/COMMON_LAYER.md` fixes the data split, label boundaries,
  smoother, scorer, and keep rule for P2-P5.
- P4's recorded results are the primary comparison: its three-seed ensemble
  reached 0.710 spellF1 / 0.906 macroF1 on Sportec validation and 0.693 / 0.904
  on the one-shot Sportec test.
- `docs/models/P4_LEARNINGS_FOR_P5.md` on the P4 branch reports that two extra
  damaged training copies added 0.051 validation spellF1, a three-seed ensemble
  added 0.025, single-seed spread was about 0.03-0.04, and four training matches
  caused fast overfitting. Those effects are at least as important as the model
  family and must be controlled.
- The same P4 note reports that 10, 30, and 60 second GRU windows tied. P5 will
  start at 30 seconds to make the first comparison exact, not because 30 seconds
  is assumed optimal.
- The closest task-specific possession study uses temporal object encoders and
  permutation-invariant aggregation, not a Transformer. It supports temporal
  modeling but does not establish attention as the winner:
  <https://doi.org/10.1016/j.eswa.2023.119780>.

### External evidence

- The canonical Transformer supplies scaled self-attention, multiple heads,
  residual blocks, and positional information; ordering must be added because
  attention alone is permutation-invariant:
  <https://arxiv.org/abs/1706.03762>.
- Simple linear models have beaten substantially more elaborate Transformers on
  several long-horizon time-series forecasting benchmarks. That is a different
  task, but it is direct evidence against assuming that more attention layers
  are automatically better for ordered numeric sequences:
  <https://arxiv.org/abs/2205.13504>.

The evidence therefore supports a **small, parameter-controlled local-global
encoder as a test**, not a large or bespoke Transformer as a default. The local
path adds an inductive bias for short continuous football motion; the attention
path tests whether longer non-local relationships help. This hybrid does not by
itself prevent match bias, so grouped evaluation remains mandatory.

## Fixed P5 v0 contract

| Item | P5 v0 decision | Reason |
|---|---|---|
| Inputs | schema `04_full_phase_context`, all `gtag_*` removed | Matches final P4 and avoids the known cross-dataset tag risk. |
| Missingness | train-only standardisation, NaN to zero, one observed flag for each sometimes-missing feature | Matches P4's GRU-D-lite boundary and IL-009. |
| Temporal mode | `offline_acausal` | The competition is offline; P5 must not be described as live. |
| Window | 10 seconds, every fifth frame, never across a period boundary | P4 measured 10/30/60 seconds as tied; 50 tokens makes the deadline run practical. Supersedes the original 30-second plan in P5-007. |
| Token | one sampled frame | Preserves per-frame labels and avoids adding patch design as a second experimental variable. |
| Position | fixed sinusoidal encoding | Supplies order without learned match-specific position parameters. |
| Local path | one depthwise temporal convolution plus a pointwise projection, residual into each frame token | Adds a small local-motion bias without a second independently trainable classifier. |
| Global path | 2 Transformer layers, `d_model=64`, 4 heads, feed-forward width 128, dropout 0.3, pre-norm | Tests non-local context while staying small enough for four training matches. |
| Head | shared linear H/A/D head at every token | Produces the common per-frame probability contract directly. |
| Loss | unweighted cross-entropy on `labels_raw`; ignored frames stay ignored | Matches P4 and the common label boundary. |
| Optimisation | AdamW 1e-3, gradient clip 1.0, early stop on validation cross-entropy | Matches P4 unless evidence requires a change. |
| Seeds | 0, 1, 2; average probabilities, then smooth once | Required by measured neural seed variance. |
| Decoder | common Viterbi penalty grid; headline also reported at penalty 20 | Keeps P2-P5 comparable. |
| Training data | common four train matches; add the same two P4-style damaged train-only copies for the fair final comparison | P4 measured this as its largest gain. |

P5 v0 must not add patching, stacked/dilated convolution blocks,
relative-position attention, class weights, a CRF, a two-head loss, pretraining,
BAR-ATH-specific logic, or a separately trained blending model. Each would
confound the first architecture comparison.

## Match-bias controls

A hybrid architecture is not accepted as a match-bias control on its own. P5
must satisfy all of these independently:

1. Model inputs contain no match, competition, provider, team, or track-slot
   identity; frame IDs remain alignment-only.
2. Splits and all early stopping stay grouped by whole match. Learned scaling
   is fit on training matches only.
3. Report each validation match separately; a mean improvement that comes from
   one match and regresses the other is not treated as broad evidence.
4. Run leave-one-training-match-out sensitivity on the frozen P5 configuration.
   This is a stability diagnostic, not another hyperparameter surface.
5. Re-score after additional label-independent ball/player masking. A gain that
   disappears under modest tracker-style variation is treated as match/damage
   shortcut evidence.
6. Run `diagnostics_round1.py` for match/damage markers and include the output
   summary in `docs/models/P5.md`.

## Work, one gate at a time

### Gate 1 — contract and synthetic tests

Deliver:

- `src/models/p5_transformer.py` with preprocessing, period-safe windows, the
  small local-global encoder, and full-frame probability stitching;
- `tests/test_p5_transformer.py` proving period isolation, missing-value
  handling, output shape, probability normalization, positional sensitivity,
  deterministic replay, and no dependence on frame IDs;
- a parameter-count assertion that keeps P5 in the same order of magnitude as
  P4 rather than winning through uncontrolled capacity.

Do not load Sportec or BAR-ATH in this gate.

### Gate 2 — one-seed pipeline proof

Deliver:

- `scripts/train_p5.py` using only `src/common/sportec.py` and the common input
  contract;
- seed 0 completes on the four training and two validation matches;
- run metadata records commit, schema, columns, temporal mode, window, step,
  parameter count, seed, epochs, validation loss, penalty curve, and confirms
  that the test match was not loaded.

The seed-0 score is a pipeline check, not a keep decision.

### Gate 3 — fair three-seed comparison

Run seeds 0, 1, and 2 with the same two train-only damaged copies used by P4.
Average probabilities, tune the common smoother on validation, and report both
raw and smoothed spellF1/macroF1 per match plus ball-visible/missing slices.

Run the match-bias controls above on the frozen configuration. They can reject
P5, but their outcomes cannot be used to tune new settings.

Compare P5 against P4 with `keep_rule` on the validation matches. Also report
the fixed penalty-20 row so the model-dependent penalty cannot hide a worse
probability model.

### Gate 4 — freeze, one-shot test, and document

Only after the Gate 3 decision is recorded:

- freeze the model and decoder settings;
- score J03WQQ once without changing P5 afterward;
- run BAR-ATH only as a report-only sanity check;
- write `docs/models/P5.md` with commands, results, failures, and the final
  keep/stop decision;
- open the P5 pull request even if the result is negative, because reproducible
  negative evidence is a deliverable.

## Keep and stop rules

1. **Keep P5 as a candidate** only if the three-seed ensemble beats P4 on
   validation smoothed spellF1 and the 95% chunk-bootstrap interval from
   `keep_rule` is entirely above zero.
2. Require the direction of the validation gain to be non-negative on both
   validation matches. A win isolated to one match fails the match-bias gate.
3. Report raw spellF1 and macroF1. A smoother-only win does not justify claims
   that attention learned a better possession representation.
4. If the interval crosses zero or is negative, record P5 as a completed
   negative result and stop. Do not respond with architecture growth or a broad
   hyperparameter sweep.
5. If P5 passes validation, the one-shot test is a transfer check, not another
   tuning surface. A test regression is reported; no setting is changed from it.
6. BAR-ATH public labels never choose a feature, model setting, seed, penalty,
   or checkpoint.

## Pre-mortem

| Likely failure | Early evidence | Mitigation |
|---|---|---|
| Attention memorises match/tracker style. | Low train loss, unstable validation seeds, or failure under extra missingness. | Small model, dropout, exact match split, damaged train copies, three seeds. |
| The hybrid hides match bias behind its local branch. | Improvement appears on only one validation match or vanishes in leave-one-match sensitivity. | Per-match keep direction, leave-one-match sensitivity, and masking stress test. |
| Position is lost or wrongly encoded. | Permuting tokens barely changes logits, or edge frames behave erratically. | Fixed positional encoding and an explicit positional-sensitivity test. |
| Comparison is unfair to P4. | Different schema, missingness flags, train copies, smoother, or seed aggregation. | Freeze the v0 contract above and report a parity checklist with every run. |
| Quadratic attention wastes compute without improving context. | Training is materially slower while the keep interval crosses zero. | Only 50 tokens in the deadline-adjusted v0; stop on a tie rather than scaling the model. |
| Validation reuse creates a false lead. | Gain is small relative to the recorded 0.03-0.04 seed spread. | Three-seed ensemble, chunk bootstrap, then one frozen test evaluation. |
