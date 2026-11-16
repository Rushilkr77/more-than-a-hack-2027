# Blend: P2 + GRU 30 s + GRU 20 s

An extra row next to P2, P3 and P4, not a new pipeline: P5 stays a separate approach. The blend averages the per-frame
P(H), P(A), P(D) of three frozen 3-seed ensembles with equal weights, then runs the common smoother once. Same split,
labels, smoother and scorer as `docs/common/COMMON_LAYER.md`.

| Member | Run folder |
|---|---|
| P2 trees, final form | `outputs/p2/h_sd_win_1_5_15_ens` |
| P4 GRU, 30 s windows, final form | `outputs/p4/04_g0_w30_s5_aug2_ens` |
| P4 GRU, 20 s windows, otherwise identical | `outputs/p4/04_g0_w20_s5_aug2_ens` |

Arithmetic mean, weights 1/3 each, switch penalty 20 (tuned on Sportec validation, common grid).

## Results

spellF1 / macroF1. BAR-ATH is the public 5-minute window, report only.

| | Sportec validation | Sportec test (J03WQQ) | BAR-ATH public 5 min |
|---|---|---|---|
| **Blend smoothed** | **0.727 / 0.912** | **0.718 / 0.911** | 0.600 / 0.852 (spellF1 interval 0.39-0.80) |
| Blend unsmoothed | 0.462 / 0.909 | 0.414 / 0.904 | 0.507 / 0.864 |
| P4 smoothed | 0.710 / 0.906 | 0.693 / 0.904 | 0.675 / 0.803 (0.47-0.86) |
| P2 smoothed | 0.668 / 0.899 | 0.662 / 0.904 | 0.735 / 0.817 (0.53-0.90) |

- Validation, blend against P4: `keep_rule` mean gain +0.020, 95% interval +0.001 to +0.038. A pass, but a narrow one.
- Test, blend against P4: +0.021, interval +0.008 to +0.036. The test match was scored once, after the configuration
  was frozen (`outputs/blend/TEST_SCORED.json`; `blend.py --final` refuses a second run).
- BAR-ATH: macroF1 rises from 0.803 to 0.852. spellF1 is 0.600 against P4's 0.675, well inside both intervals. The
  blend predicts 11 spells against 17 true there, so it merges possessions on this clip.

## Gate

Fixed before any run. A candidate replaces the incumbent only if both hold:

1. Sportec validation: the `keep_rule` 95% interval on smoothed output is above 0. A tie keeps the simpler option.
2. BAR-ATH guard: no clear drop against the incumbent (macroF1 by more than about 0.05, or spellF1 below the
   incumbent's bootstrap lower bound).

BAR-ATH only vetoes a collapse. It never chose a member, weight, penalty or seed (RULES.md rule 2).

## Decision log

Validation spellF1 / macroF1, tuned penalty, BAR-ATH smoothed spellF1 / macroF1. 24 candidates, all on validation.

### Stage 1 and 2: members (incumbent at start: P4 alone, 0.710)

| Candidate | Validation | Pen | BAR-ATH | Against incumbent | Verdict |
|---|---|---|---|---|---|
| P2 + GRU 30 s | 0.705 / 0.908 | 10 | 0.650 / 0.828 | +0.010 (-0.009, +0.029) | tie |
| GRU 30 s + GRU 20 s | 0.715 / 0.909 | 20 | 0.614 / 0.885 | crosses 0 | tie |
| P2 on damaged copies + GRU 30 s | 0.709 / 0.910 | 10 | 0.650 / 0.809 | crosses 0 | tie |
| **P2 + GRU 30 s + GRU 20 s** | **0.727 / 0.912** | 20 | 0.600 / 0.852 | **+0.020 (+0.001, +0.038)** | **pass, new incumbent** |
| P2 on damaged copies + GRU 30 s + GRU 20 s | 0.731 / 0.912 | 20 | 0.600 / 0.847 | +0.001 (-0.007, +0.008) | tie, plain P2 is simpler |
| P2 with 10 s rolling window, damaged copies + both GRUs | 0.725 / 0.912 | 20 | 0.600 / 0.798 | crosses 0 | tie |
| P2 5 seeds, damaged copies + both GRUs | 0.729 / 0.912 | 20 | 0.600 / 0.849 | crosses 0 | tie |
| + GRU 10 s (4 members) | 0.731 / 0.912 | 20 | 0.747 / 0.848 | crosses 0 | tie |
| + GRU 60 s (4 members) | 0.724 / 0.912 | 20 | 0.600 / 0.830 | crosses 0 | tie |
| + GRU 10 s + GRU 60 s (5 members) | 0.729 / 0.912 | 20 | 0.600 / 0.831 | crosses 0 | tie |
| GRU 30 s with 5 seeds in the blend | 0.728 / 0.912 | 20 | 0.627 / 0.882 | crosses 0 | tie |
| P2 + GRU 30 s + GRU 10 s (10 s in place of 20 s) | 0.728 / 0.911 | 20 | 0.747 / 0.842 | crosses 0 | tie |
| P2 + GRU 30 s + GRU 60 s | 0.717 / 0.911 | 20 | 0.600 / 0.797 | crosses 0 | tie |

Single members for reference: GRU 20 s 0.701, GRU 10 s 0.726, GRU 60 s 0.699, GRU 30 s with 5 seeds 0.710, P2 on
damaged copies 0.677, P2 on damaged copies with 5 seeds 0.683. GRU 10 s against GRU 30 s alone is a tie
(+0.011, interval -0.009 to +0.032).

### Stage 3: weights and mean (on the incumbent)

| Candidate | Validation | Pen | BAR-ATH | Verdict |
|---|---|---|---|---|
| GRU family weight 0.25 | 0.696 / 0.908 | 10 | 0.656 / 0.865 | tie (lower) |
| GRU family weight 0.50 | 0.717 / 0.911 | 10 | 0.650 / 0.857 | tie (lower) |
| GRU family weight 0.75 | 0.734 / 0.912 | 20 | 0.600 / 0.853 | tie |
| Geometric mean | 0.730 / 0.912 | 20 | 0.600 / 0.851 | tie |

Equal weights (GRU family 0.67) stay.

### Stage 4: smoother (last resort, same common smoother with its optional dials)

| Candidate | Validation | Pen | BAR-ATH | Verdict |
|---|---|---|---|---|
| Finer grid 10 / 15 / 20 / 25 / 30 | 0.727 at 20 (15: 0.726, 25: 0.716) | 20 | unchanged | no change |
| live↔dead penalty 10 / 40 | 0.728 / 0.724 | 20 | 0.600 / 0.852 | tie |
| team↔team penalty 10 / 40 | 0.718 / 0.724 | 10 / 5 | 0.650 / 0.863, 0.606 / 0.868 | tie |
| minimum spell 1 s / 2 s | 0.727 / 0.721 | 20 / 10 | 0.600 / 0.852, 0.650 / 0.864 | tie |

No dial moved the score, so no new smoothing algorithm was built. The headline uses the common smoother unchanged.

## What this says

- The gain comes from mixing three different views: trees, a GRU with 30 s context and a GRU with 20 s context. No
  pair of them passes; the three together do.
- A second window length helps only as a blend member. Alone, 20 s scores below 30 s.
- Extra seeds, extra damaged copies for P2, a 10 s rolling window for P2, weights, the mean type and every smoother
  dial are ties. The remaining errors are not something averaging or smoothing fixes.
- Swapping the 20 s member for the 10 s one ties on Sportec and reads higher on BAR-ATH (0.747). It was not taken:
  the 20 s member was the planned candidate and passed first, and choosing between ties on BAR-ATH is tuning on it.

## Caveats

- 24 validation decisions on top of P2's about 30 and P4's about 10. The validation pass is narrow; the test match
  agrees (+0.025 spellF1), which is the cleaner evidence.
- The blend needs three models at inference instead of one.
- BAR-ATH has 17 possessions; its numbers are a sanity check only.

## Reproduce

    python scripts/train_p2.py --ensemble --export
    python scripts/train_p4.py --seed 0 --seed 1 --seed 2 && python scripts/train_p4.py --ensemble --export
    python scripts/train_p4.py --window 20 --seed 0 --seed 1 --seed 2
    python scripts/train_p4.py --window 20 --ensemble --export
    python scripts/blend.py --name p2_p4w30_p4w20 --members outputs/p2/h_sd_win_1_5_15_ens \
        outputs/p4/04_g0_w30_s5_aug2_ens outputs/p4/04_g0_w20_s5_aug2_ens
    python scripts/blend.py --compare outputs/p4/04_g0_w30_s5_aug2_ens outputs/blend/p2_p4w30_p4w20

`--export` saves test-match and BAR-ATH probabilities without scoring the test match. Logs of every candidate are in
`outputs/blend/`.
