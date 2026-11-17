# Best-model audit and bounded experiment log

Date: 2026-11-17

## Objective

Select the strongest defensible possession model across approaches. No model
family is privileged. A candidate must use the common processed-data contract,
the fixed whole-match split, the common scorer and the chunk-bootstrap keep
rule. The already-scored Sportec test and BAR-ATH public window are report-only.

## Existing hybrid PR 16

PR 16 reports an equal arithmetic blend of P2, P4 with 30-second windows and
P4 with 20-second windows at 0.727 validation spellF1 and 0.718 on its one-shot
test. It already tried 24 validation candidates: global weights, arithmetic and
geometric means, extra seeds, damaged P2 copies, P2 rolling-window changes,
GRU windows 10/20/30/60 seconds, and common-smoother dials. Those choices will
not be repeated.

The original probabilities are git-ignored and unavailable. Audit therefore
regenerates every member before testing anything new.

## Environment and reproducibility findings

- The branch declares Python-compatible dependencies only partially. Its P2
  code calls a scikit-learn >=1.7 API but the existing Python 3.9 environment
  resolves scikit-learn 1.6.1. A recorded compatibility fallback is required.
- Importing `scripts.train_p2` under a clean environment could not find its
  sibling augmentation module. Adding the repository `scripts/` directory to
  the import path fixes the package/script mismatch.
- A clean audit environment uses Python 3.12, scikit-learn 1.7.2 and the
  branch-declared Torch 2.14.1.
- The deterministic P4 damaged copies reuse the shared builder outputs with the
  exact documented seeds and parameters; validation is not augmented.

### Regenerated validation results so far

| Member | Environment | SpellF1 | MacroF1 | Published spellF1 |
|---|---|---:|---:|---:|
| P2 three-seed ensemble | Python 3.12 / sklearn 1.7.2 | 0.686 | 0.904 | 0.668 |
| P4 30-second three-seed ensemble | Torch 2.14.1 | 0.695 | 0.906 | 0.710 |
| P3 schema-04 10s/10s | Torch 2.15.0 MPS | 0.655 | 0.886 | 0.630 |

The published scores are not exactly reproducible from the PR and its declared
requirements. Fresh probabilities, not the published point estimates, are the
authoritative basis for new comparisons.

## New experiment gate

### E1 — add the selected P3 TCN

Hypothesis: P3's class-weighted asymmetric TCN has complementary temporal
errors to trees and bidirectional GRUs. On both validation matches, its raw
frame-error correlation with P2 is only 0.56-0.59 and it recovers 2.5%-3.1% of
all frames that P2 misses. The broader TCN literature supports hierarchical
temporal convolutions for segment duration and long-range dependencies, but
does not establish that P3 will improve this task:
<https://openaccess.thecvf.com/content_cvpr_2017/html/Lea_Temporal_Convolutional_Networks_CVPR_2017_paper.html>.

Protocol, fixed before blending:

1. Regenerate P2, P4-30 and P4-20 under their declared environments.
2. Regenerate P3's documented winning schema-04 10-second-past / 10-second-
   future configuration and export H/A/D probabilities.
3. Rebuild the three-member incumbent with equal arithmetic weights and one
   common smoother.
4. Add P3 as a fourth equal-weight member. Do not search P3 weights, member
   subsets, means or penalties outside the common grid.
5. Keep only if the chunk-bootstrap interval over the incumbent is wholly
   positive and per-match direction is non-negative on both validation matches.

Result:

| Candidate | SpellF1 / macroF1 | J03WMX | J03WOY | Chunk gain (95% CI) | Keep |
|---|---:|---:|---:|---:|---|
| Regenerated P2 + P4-30 + P4-20 | 0.715 / 0.911 | 0.676 | 0.754 | incumbent | yes |
| + P3, four equal members | **0.728 / 0.912** | 0.696 | 0.761 | +0.002 [-0.026, +0.025] | **no** |

The P3 point estimate is the strongest freshly generated validation score, and
the aggregate spellF1 direction is positive on both matches. The 97-chunk
bootstrap crosses zero, however, and mean chunk direction is slightly negative
on J03WOY (-0.0003). P3 is therefore recorded but not promoted. No P3 weight or
member subset was tried.

### E2 — duration-aware structured decoder, only if E1 fails

Current models optimize frame cross-entropy while the primary metric scores
possession spells. PR 16 only varies scalar transition penalties and minimum
spell thresholds. A semi-Markov decoder with class-specific duration priors
estimated from training labels is a materially different hypothesis: it models
segment duration directly without fitting a new dial to validation.

This is supported by two direct structured-sequence results, while remaining a
hypothesis for football possession:

- semi-Markov CRFs assign labels to variable-length segments and permit segment
  features with exact polynomial-time inference:
  <https://proceedings.neurips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html>;
- MS-TCN adds a smoothing loss specifically to reduce over-segmentation in
  framewise temporal models:
  <https://openaccess.thecvf.com/content_CVPR_2019/papers/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.pdf>.

The four training matches also show a concrete duration mismatch that one
scalar switch penalty cannot express: raw-label median spells are 3.84 s for H,
3.60 s for A and 21.20 s for D; their 90th percentiles are 23.49, 14.44 and
52.42 s respectively. These are descriptive training-only statistics, not
validation-selected cutoffs.

E2 may run only after E1 is recorded. Its duration distribution must be fitted
from training matches alone, and it must face the same keep and per-match gates.

Result:

| Candidate | SpellF1 / macroF1 | J03WMX | J03WOY | Chunk gain (95% CI) | Keep |
|---|---:|---:|---:|---:|---|
| Three-member incumbent | 0.715 / 0.911 | 0.676 | 0.754 | incumbent | yes |
| Training-only semi-Markov decoder | 0.714 / 0.910 | 0.674 | 0.754 | +0.0016 [-0.0043, +0.0099] | **no** |

The structured decoder is rejected. It lowers aggregate spellF1 and its
97-chunk bootstrap crosses zero. This result also rules out further duration-
decoder tuning on the reused validation matches.

### E3 — MS-TCN truncated temporal smoothing loss

E1 shows that the TCN supplies complementary errors but is individually weak;
E2 shows that changing only the final decoder does not fix the sequence. The
next hypothesis therefore acts during representation learning. MS-TCN combines
classification loss with a truncated MSE between adjacent log probabilities to
penalize over-segmentation while preserving confident boundaries. The paper's
fixed defaults are weight 0.15 and truncation threshold 4:
<https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html>.

Protocol, fixed before training:

1. Keep P3's selected schema 04, 10-second past/future context, architecture,
   seed 2027, augmentation, optimiser and early stopping unchanged.
2. Add only the paper's truncated temporal smoothing term with weight 0.15 and
   threshold 4. Do not sweep either constant.
3. Train one run and export validation probabilities through the same processed
   input contract. The held-out test and BAR-ATH are not loaded.
4. Add this run as the fourth equal member of the regenerated three-member
   incumbent, tune only over the common pre-existing penalty grid, and compare
   with the same 97-chunk keep rule.
5. Promote only if the keep interval is wholly positive and both validation
   matches have non-negative mean chunk direction. Otherwise retain it only as
   an experimental result. Do not try a second smoothing constant or seed.

Result:

| Candidate | SpellF1 / macroF1 | J03WMX | J03WOY | Chunk gain vs incumbent (95% CI) | Keep |
|---|---:|---:|---:|---:|---|
| Original P3 four-member hybrid | **0.7282 / 0.9121** | 0.6957 | 0.7607 | +0.0022 [-0.0263, +0.0249] | no |
| T-MSE P3 four-member hybrid | 0.7277 / 0.9113 | 0.7060 | 0.7494 | +0.0026 [-0.0273, +0.0270] | **no** |

T-MSE raises P3's unsmoothed spellF1 but lowers its smoothed score from 0.6547
to 0.6456. Its hybrid is 0.0005 below the original P3 point leader, its
bootstrap crosses zero, and its J03WOY mean chunk direction is negative
(-0.0095). It is rejected. No second constant or seed is run.

### E4 — visibility-gated P3 hybrid

P4's documented weak slice is live play with a missing ball, while P3 is the
only member that retains the smart-tagging channels designed to reconstruct
phase context. A match-agnostic reliability gate can therefore restrict P3 to
the frames where its complementary inputs are most relevant. This avoids the
user-rejected strategy of biasing weights by match identity.

Protocol, fixed before scoring:

1. On `ball_visible == true`, use the regenerated equal three-member incumbent
   (P2 + P4-30 + P4-20).
2. On `ball_visible == false`, use the equal four-member blend that adds the
   original selected P3. Use the processed bundle's existing boolean flag; do
   not alter or redamage validation data.
3. Decode once with the same common smoother and pre-existing penalty grid.
4. Compare against both the three-member incumbent and the 0.7282 full-P3
   point leader. It is a new best only if it exceeds 0.7282, has non-negative
   match-level aggregate and chunk direction on both matches, and passes the
   97-chunk interval against the incumbent. No visibility threshold or soft
   transition is tuned.

Result:

| Candidate | SpellF1 / macroF1 | J03WMX | J03WOY | Keep vs incumbent | Keep vs full P3 |
|---|---:|---:|---:|---:|---:|
| Full P3 hybrid | **0.7282 / 0.9121** | 0.6957 | 0.7607 | no | leader |
| Visibility-gated P3 | 0.7249 / **0.9122** | 0.6872 | 0.7625 | no | no |

The gate is rejected. It is 0.0034 below the full-P3 point leader; its interval
crosses zero versus both the incumbent and full-P3 hybrid. No threshold or
soft-transition follow-up is run.

## Deadline selection

The equal four-member P2 + P4-30 + P4-20 + original P3 hybrid is the strongest
freshly measured candidate at 0.7282 spellF1. It also improves aggregate
spellF1 on both validation matches and raises macroF1 from 0.9111 to 0.9121.
Because the user explicitly prefers results over a maximally conservative
choice for the deadline, it is packaged as the deadline point-estimate leader
in `configs/best_model_selection.json`.

This is not a claim of statistically established superiority. Its 97-chunk
interval against the three-member incumbent crosses zero, the validation set
has been reused, and the previously scored test cannot provide an independent
confirmation. The defensible conservative incumbent remains the three-member
hybrid at 0.7151. A new independent match is the blocking evidence needed to
resolve that uncertainty honestly.

### E5 — match-cross-fitted linear stacking

Equal averaging assumes every member and class are equally reliable. Learning
weights on the same validation frames used for scoring would be leakage, while
global manual weight searches were already exhausted by PR 16. Stacked
generalization instead learns a second-level model from held-out base-model
predictions:
<https://doi.org/10.1016/S0893-6080(05)80023-1>.

Protocol, fixed before scoring:

1. Use only the four frozen validation probability tensors from P2, P4-30,
   P4-20 and the original P3. Features are the 12 clipped log probabilities;
   no match ID, frame index, label-derived feature or test input is allowed.
2. Fit a standardized multinomial logistic stacker with fixed L2 `C=1`, LBFGS,
   no class reweighting and no hyperparameter search.
3. Fit on J03WMX adjusted labels and predict J03WOY; fit on J03WOY and predict
   J03WMX. Thus every scored frame comes from a stacker that did not see that
   match's labels. Fit a final deployment artifact on both only after cross-fit
   predictions are frozen; do not score that in-sample artifact.
4. Apply the common smoother once and select only from its existing penalty
   grid. Compare the cross-fitted output against both the 0.7151 conservative
   incumbent and 0.7282 equal-four-member point leader with the same 97-chunk
   keep rule and per-match direction checks.
5. Promote only if it exceeds 0.7282, improves both match aggregates, has
   non-negative mean chunk direction on both, and passes the interval against
   the incumbent. Do not tune `C`, feature subsets, sampling, class weights or
   member sets after seeing the result.

Result:

| Candidate | SpellF1 / macroF1 | J03WMX | J03WOY | Mean chunk gain vs full P3 | Keep |
|---|---:|---:|---:|---:|---|
| Equal four-member P3 hybrid | **0.7282 / 0.9121** | 0.6957 | 0.7607 | leader | yes |
| Match-cross-fitted stacker | 0.6997 / 0.9063 | 0.6731 | 0.7262 | -0.0151 [-0.0399, +0.0148] | **no** |

The learned reliability weights do not transfer between the two validation
matches. The stacker also loses to the conservative three-member incumbent,
with a negative J03WOY mean chunk gain of -0.0273. It is rejected; the final
two-match artifact is retained only for reproducibility and must not be called
the selected model. No stacker setting is tuned.

### E6 — learned boundary costs on the four-member point leader

The sibling `codex/possession-boundary` branch already tested a training-only
boundary detector on the regenerated three-member hybrid. With identical
probabilities it raised spellF1 from 0.7151 to 0.7173, improved both whole
matches, but failed the keep interval. A restart-segment branch made no edits.
Those experiments are existing evidence and will not be repeated or tuned.

The remaining untested interaction is whether P3's complementary probabilities
and input-dependent switch costs help together. The detector follows the
boundary-aware temporal segmentation motivation of Ishikawa et al.:
<https://arxiv.org/abs/2007.06866>.

Protocol, fixed before scoring:

1. Import the exact `codex/possession-boundary` v1 implementation: schema-03
   team-symmetric tracking features, ±0.4-second context, targets within ±0.2
   seconds of a raw-label change, every fifth training frame, and the fixed
   120-iteration histogram tree. Do not change its columns or parameters.
2. Train only on the four training matches. Use the frozen P2, P4-30, P4-20
   and original P3 probabilities in an equal arithmetic blend.
3. Decode both baseline and candidate at fixed penalty 20. Candidate switch
   cost is the detector's training-prior log-odds correction bounded to ±10;
   there is no validation-selected threshold or new penalty search.
4. Compare against the same four-member fixed decoder. Promote only if the
   candidate exceeds 0.7282, improves aggregate and mean chunk direction on
   both matches, and its 97-chunk interval is wholly positive. Do not load the
   held-out test or BAR-ATH and do not tune after the result.

## Test-set boundary

No new candidate in this log may be selected from J03WQQ or BAR-ATH. Even if a
candidate passes validation, those previously viewed sets can only be reported
as non-independent context. A new independent match would be required for a
clean final generalization claim.
