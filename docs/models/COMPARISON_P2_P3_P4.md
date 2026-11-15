# P2 vs P3 vs P4: comparison report

Three approaches, one protocol. Every row below is a 3-seed probability ensemble decoded by the common Viterbi smoother
at penalty 20, scored with the common scorer. spellF1 is the headline, macroF1 the frame-level companion.

| | P2 | P3 | P4 |
|---|---|---|---|
| Model | gradient-boosted trees (`HistGradientBoosting`) | causal TCN (`pytorch-tcn`), 10 s past / 10 s future | bidirectional GRU, 30 s windows |
| Inputs | schema 04 **without** `gtag_*`, plus past/next 1, 5, 15 s rolling mean/std/diff | schema 04 **with** `gtag_*`, raw columns only | schema 04 **without** `gtag_*`, plus observed-flag columns |
| Training data | common damaged bundles, mirrored | common damaged bundles, mirrored | common bundles **plus 2 P4-local damaged copies** of each training match, mirrored |
| Loss | cross-entropy | class-weighted cross-entropy | cross-entropy |
| Code | `scripts/train_p2.py --ensemble` | PR 13 (`configs/p3_tcn_10x10.json`) | `scripts/train_p4.py --ensemble` (PR 14) |

## Results

spellF1 / macroF1. Sportec validation = the 2 validation matches (mean). Sportec test = J03WQQ (294 true possessions).
BAR-ATH = the public 5-minute window (17 true possessions), report only.

### Smoothed (common smoother, penalty 20)

| | Sportec validation | Sportec test (J03WQQ) | BAR-ATH public 5 min |
|---|---|---|---|
| **P2** | 0.668 / 0.899 | 0.662 / 0.904 | 0.735 / 0.817 |
| **P3** | 0.637 / 0.883 | 0.652 / 0.884 | 0.593 / 0.844 |
| **P4** | **0.710 / 0.906** | **0.693 / 0.904** | 0.675 / 0.803 |
| Baseline smooth50 (organiser) | 0.219 / 0.449 | 0.223 / 0.461 | 0.326 / 0.621 |

### Unsmoothed (argmax of the ensemble probabilities)

| | Sportec validation | Sportec test (J03WQQ) | BAR-ATH public 5 min |
|---|---|---|---|
| **P2** | 0.363 / 0.900 | 0.345 / 0.899 | 0.380 / 0.844 |
| **P3** | 0.361 / 0.883 | **0.363** / 0.881 | 0.467 / 0.854 |
| **P4** | **0.440 / 0.904** | 0.355 / 0.898 | **0.644** / 0.808 |
| Baseline raw (organiser) | 0.049 / 0.430 | 0.042 / 0.428 | 0.039 / 0.597 |

### Control and spread

| | Sportec validation | Sportec test | BAR-ATH |
|---|---|---|---|
| P2 trained on P4's augmented copies, smoothed | 0.677 / 0.905 | 0.679 / 0.906 | 0.674 / 0.846 |
| P2 trained on P4's augmented copies, unsmoothed | 0.385 / 0.904 | 0.373 / 0.903 | 0.316 / 0.835 |
| P3 single seeds 0 / 1 / 2, smoothed spellF1 | 0.597 / 0.634 / 0.622 | 0.640 / 0.630 / 0.655 | 0.600 / 0.824 / 0.787 |

The P2-with-P4's-copies row is a diagnostic, not an official entry: it separates what P4's training data contributes from
what its GRU contributes.

## How the three were kept aligned

Everything that decides the ranking is shared (`docs/common/COMMON_LAYER.md`); only the model is per approach.

- **Data:** the same damaged Sportec bundles (`scripts/build_sportec_bundles.py`, fixed seeds), loaded through
  `sportec.load` and `sportec.classification_inputs`, one leak-safe input contract for all three.
- **Split:** `configs/sportec_split.json`, 4 train / 2 validate / 1 test, fixed. The test match is never loaded by
  training code.
- **Labels:** `labels_raw` to train, `labels_adjusted` to tune and score. Home/away mirrored copies in training.
- **Smoother:** `src/decoder/viterbi.py`, 3-state, one switch penalty. The headline penalty is **20 for all three**
  (a value on the common `PENALTIES` grid, tuned on Sportec validation). Per-model tuned penalties differ run to run
  (P3 seeds tuned to 5, 20, 20 and its ensemble to 10); fixing 20 keeps the comparison like for like.
- **Scorer:** `sportec.score`, spellF1 and macroF1 averaged per match for validation.
- **Seeds:** 3-seed probability ensemble for every approach (seeds 0, 1, 2): average P(H), P(A), P(D) across seeds,
  then smooth once. A single seed can differ from the ensemble by 0.01 to 0.05 on spellF1.
- **BAR-ATH:** inference only. Never used to choose a feature set, window, penalty or seed.
- **Test match:** reported only after each model was frozen.

What stays different on purpose: the model, its extra features, its context window, and (for P4) its extra damaged
training copies. P3 keeps `gtag_*` because it helped P3 (schema 03 was clearly worse); P2 and P4 drop it.

## Where each approach is better or worse

**P4 (GRU)**
- Better: best smoothed spellF1 on Sportec validation (+0.042 over P2) and test (+0.031); best unsmoothed validation
  (0.440); best unsmoothed BAR-ATH (0.644, 19 predicted spells against 17 true).
- Worse: lowest BAR-ATH smoothed macroF1 (0.803) and a lower smoothed spellF1 than P2 or P3's best seeds there.
  Smoothing helps it little on BAR-ATH.
- Caveat: a large part of its Sportec lead is the extra damaged training copies (+0.051 in its own ablation). With the
  same copies P2 reaches 0.679 on test, so P4's lead there shrinks to +0.014, within seed noise. On validation the
  lead stays +0.033.

**P2 (trees)**
- Better: ties P4 on Sportec macroF1 (0.904 on test); cheapest and simplest to run; numerically highest smoothed spellF1 on
  BAR-ATH (0.735 vs P4 0.675 and P3 0.593), though that gap is within noise.
- Worse: behind P4 on Sportec spellF1; weakest unsmoothed on BAR-ATH (0.380), so it leans on the smoother.
- Note: `gtag_*` helps it on Sportec (+0.03) but `gtag_poss` costs about 0.09 on BAR-ATH, so it stays out.

**P3 (TCN)**
- Better: BAR-ATH macroF1 is competitive (0.844, single seeds 0.83 to 0.91); highest unsmoothed test spellF1 by a hair
  (0.363); BAR-ATH unsmoothed 0.467, above P2.
- Worse: lowest smoothed spellF1 and macroF1 on both Sportec sets (macroF1 about 0.02 below P2 and P4); heaviest to
  train; very unstable on BAR-ATH smoothed (single seeds 0.60 to 0.82, ensemble 0.593).
- Note: longer context helped it (3 s / 1 s window about 0.61 on validation, 10 s / 10 s 0.64).

## Reading the table

1. **Sportec ranking:** P4 > P2 > P3 on smoothed spellF1 for validation and test. P4 over P2 is a clear gap on
   validation and a small one on test. P2 over P3 is clear on validation and within noise on test.
2. **BAR-ATH cannot rank them.** 17 possessions put a 95% interval of about 0.5 to 0.9 on each smoothed spellF1, and
   P3's own seeds span 0.60 to 0.82. Treat BAR-ATH as a sanity check that nothing collapsed.
3. **Smoothing matters most for P2 and P3** (P2 gains about 0.3 spellF1, P3 about 0.3); P4 gains the least on BAR-ATH.
4. **Frame-level accuracy (macroF1):** P2 and P4 tie at about 0.90 on Sportec; P3 is about 0.88. All three are far above
   the baseline (0.45 to 0.46).

## Caveats and provenance

- Validation was reused for tuning: about 30 decisions for P2, about 10 for P4 (keep-rule ablations), not recorded for P3.
  The validation column is optimistic; the test column is the cleaner read.
- The Sportec test match J03WQQ has been scored more than once for P2 (single seeds, the ensemble, and the augmented
  control), and once each for P3 and P4. It was never used to choose a model setting, but make no new decisions on it.
- P4's rows are taken from `docs/models/P4.md` and were produced by `scripts/train_p4.py`. P2's validation row is
  reproducible with `python scripts/train_p2.py --ensemble`. P2's test and BAR-ATH rows, P3's 3-seed rows and the
  augmented-P2 control came from one-off scoring scripts that are not in the repository.
- P3's own PR reports single-seed numbers (seed 2027: 0.630 validation, 0.647 test, 0.680 BAR-ATH); they fall inside
  the seed range above.
- Penalty 20 is the common-grid headline for all three. Own smoother experiments are welcome as extra rows only.
