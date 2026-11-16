# P2 vs P3 vs P4 vs blend: comparison report

Extends `docs/models/COMPARISON_P2_P3_P4.md` (PR 14) with the blend from `docs/models/BLEND.md`. The P2, P3 and P4
rows are copied from that report unchanged; only the blend column is new. Same protocol for every column: 3-seed
probability ensembles, the common Viterbi smoother at penalty 20, the common scorer. spellF1 is the headline, macroF1
the frame-level companion.

| | P2 | P3 | P4 | Blend |
|---|---|---|---|---|
| Model | gradient-boosted trees | causal TCN, 10 s past / 10 s future | bidirectional GRU, 30 s windows | equal-weight average of P2, P4 (30 s) and a second P4 GRU with 20 s windows |
| Inputs | schema 04 without `gtag_*`, plus rolling 1, 5, 15 s stats | schema 04 with `gtag_*` | schema 04 without `gtag_*`, plus observed flags | its members' inputs, unchanged |
| Training data | common damaged bundles, mirrored | common damaged bundles, mirrored | common bundles plus 2 P4-local damaged copies, mirrored | nothing trained; members are frozen |
| Models at inference | 3 | 3 | 3 | 9 (three 3-seed ensembles) |
| Code | `scripts/train_p2.py --ensemble` | PR 13 | `scripts/train_p4.py --ensemble` | `scripts/blend.py` |

## Results

spellF1 / macroF1. Sportec validation = the 2 validation matches (mean). Sportec test = J03WQQ (294 true possessions).
BAR-ATH = the public 5-minute window (17 true possessions), report only.

### Smoothed (common smoother, penalty 20)

| | Sportec validation | Sportec test (J03WQQ) | BAR-ATH public 5 min |
|---|---|---|---|
| **P2** | 0.668 / 0.899 | 0.662 / 0.904 | **0.735** / 0.817 |
| **P3** | 0.637 / 0.883 | 0.652 / 0.884 | 0.593 / 0.844 |
| **P4** | 0.710 / 0.906 | 0.693 / 0.904 | 0.675 / 0.803 |
| **Blend** | **0.727 / 0.912** | **0.718 / 0.911** | 0.600 / **0.852** |
| Baseline smooth50 (organiser) | 0.219 / 0.449 | 0.223 / 0.461 | 0.326 / 0.621 |

### Unsmoothed (argmax of the ensemble probabilities)

| | Sportec validation | Sportec test (J03WQQ) | BAR-ATH public 5 min |
|---|---|---|---|
| **P2** | 0.363 / 0.900 | 0.345 / 0.899 | 0.380 / 0.844 |
| **P3** | 0.361 / 0.883 | 0.363 / 0.881 | 0.467 / 0.854 |
| **P4** | 0.440 / 0.904 | 0.355 / 0.898 | **0.644** / 0.808 |
| **Blend** | **0.462 / 0.909** | **0.414 / 0.904** | 0.507 / **0.864** |
| Baseline raw (organiser) | 0.049 / 0.430 | 0.042 / 0.428 | 0.039 / 0.597 |

### What smoothing adds (spellF1, smoothed minus unsmoothed)

| | Sportec validation | Sportec test | BAR-ATH |
|---|---|---|---|
| P2 | +0.305 | +0.317 | +0.355 |
| P3 | +0.276 | +0.289 | +0.126 |
| P4 | +0.270 | +0.338 | +0.031 |
| Blend | +0.265 | +0.304 | +0.093 |

### Significance and spell counts

| | Result |
|---|---|
| Blend against P4, Sportec validation (`keep_rule`) | +0.020, 95% interval +0.001 to +0.038: pass, narrow |
| Blend against P4, Sportec test (`keep_rule`) | +0.021, interval +0.008 to +0.036: pass |
| Blend against P2, Sportec validation (`keep_rule`) | +0.042, interval +0.020 to +0.067: pass |
| BAR-ATH smoothed spellF1 intervals | P2 0.53-0.90, P4 0.47-0.86, blend 0.39-0.80: all overlap |
| Predicted spells on Sportec test (294 true) | P4 267, blend 269 |
| Predicted spells on BAR-ATH window (17 true) | P2 14, P4 15, blend 11 |
| Live spells over the whole BAR-ATH match (no labels, penalty 20) | P2 315, P4 273, blend 260 |

## Reading the table

1. **Sportec ranking: blend > P4 > P2 > P3** on smoothed spellF1, for validation and test. The blend's lead over P4 is
   small but passes the keep rule on both sets, and the test match was scored once after the blend was frozen.
2. **Frame level:** the blend has the highest macroF1 on all three datasets (0.91 on Sportec, 0.852 on BAR-ATH).
3. **BAR-ATH still cannot rank them.** The blend has the best macroF1 and the second-lowest smoothed spellF1 there, with
   intervals that overlap every other approach. One merged possession moves spellF1 by about 0.05.
4. **The blend merges possessions on the BAR-ATH window** (11 predicted, 17 true). Over the whole match its spell
   count is 5% below P4's, so this is mostly the particular five minutes, not a collapse. All approaches predict
   fewer, longer possessions than the truth; that risk is shared.
5. **Smoothing matters less for the blend on BAR-ATH** (+0.09) than on Sportec (+0.27 to +0.30). Its unsmoothed
   output there is already close in spell count (24 against 17), and penalty 20 over-merges it. The same blend at
   penalty 10 ties on Sportec validation (0.718) and gives 15 spells on the window. It was not chosen: picking a
   penalty from BAR-ATH is tuning on it.

## Where each stands

- **Blend:** best on Sportec, best frame-level everywhere. Cost: three models' worth of inference and a narrow
  validation margin. Gain comes only from the three-way mix; no pair of members passes.
- **P4:** best single approach on Sportec. Simpler to explain and run than the blend.
- **P2:** cheapest. Highest BAR-ATH smoothed spellF1, inside the noise.
- **P3:** not in the blend because PR 13 does not export probabilities. It would join with no blend-code change once
  it writes `proba_<match>.npy`.

## Caveats and provenance

- Everything in "Caveats and provenance" of `COMPARISON_P2_P3_P4.md` still applies to the P2, P3 and P4 columns.
- The blend used 24 further validation decisions (`BLEND.md` decision log). Its validation column is optimistic; the
  test column is the cleaner read.
- The Sportec test match was scored exactly once for the blend (`outputs/blend/TEST_SCORED.json`). Make no new
  decisions on it.
- Blend rows are reproducible with the commands at the end of `BLEND.md`. P2 and P4 reran to identical numbers on this
  branch before blending.
