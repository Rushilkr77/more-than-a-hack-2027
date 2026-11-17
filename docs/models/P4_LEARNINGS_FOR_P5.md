# P4 learnings for P5

Findings from building P4 (bidirectional GRU, `docs/models/P4.md`) that later approaches can use. These are data and
process findings, not P4 choices: P5 tunes its own settings with `keep_rule` on its own Sportec validation results.
Same split, labels, smoother and scorer as `docs/common/COMMON_LAYER.md`.

## Numbers to beat

P4 = 3-seed probability ensemble `04_g0_w30_s5_aug2_ens`, penalty 20. spellF1 / macroF1.

| | Sportec validation (2 matches) | Sportec test (J03WQQ) | BAR-ATH public 5 min (report only) |
|---|---|---|---|
| P4 smoothed | 0.710 / 0.906 | 0.693 / 0.904 | 0.675 / 0.803 (spellF1 interval 0.47-0.86) |
| P4 unsmoothed | 0.440 / 0.904 | 0.355 / 0.898 | 0.644 / 0.808 |
| P2 ensemble smoothed | 0.668 / 0.899 | 0.662 / 0.904 | 0.735 / 0.817 |

Compare against these as a 3-seed ensemble of your own. The test match has been scored for P4; do not tune on it.

## What moved the score

1. **Extra damaged training copies were the biggest lever.** Two extra damaged copies of each training match:
   +0.051 on schema 04 and +0.041 on schema 03, both beyond noise. The gain is largest where the ball is missing
   (live-play accuracy 0.81 -> 0.84, dead-ball precision 0.91 -> 0.93 on missing-ball frames).
   - Two different damage patterns together (copy 1: about 42% hidden, long gaps; copy 2: about 50% hidden, short gaps)
     beat either one alone. Four copies added nothing over two.
   - How P4 builds them (`scripts/build_p4_aug.py`): the common damage code and `RANGES` from
     `scripts/build_sportec_bundles.py`, rerun on training matches only with new fixed seeds. Gaps are placed without
     labels and nothing is fitted to BAR-ATH. Validation and test are never copied, so every pipeline is still scored
     on the one common damaged version.
2. **A 3-seed probability ensemble adds +0.025 beyond noise** (average P(H), P(A), P(D), then tune the penalty and
   smooth once). P2 and P4 both submit this form.
3. **Schema 04 beat 03 beyond noise for a sequence model** once the damaged copies were in (-0.028 for 03, interval
   -0.044 to -0.012). The difference is the 12 team-dynamics columns. `gtag_*` was a tie on Sportec (-0.011, interval
   crosses 0), so P4 dropped it as the simpler option. P2 found `gtag_poss` costs about 0.09 on BAR-ATH.

## What did not matter

4. Window length (10 / 30 / 60 s) and frame step (2 vs 5) were all ties for the GRU. A smaller GRU (hidden 32 vs 64)
   lost beyond noise.

## Noise and measurement

5. **Seed spread for a neural model is large:** single seeds spread about 0.03-0.04 on Sportec validation, against
   about 0.01 for P2's trees. Run 3 seeds before reading any difference.
6. **The public BAR-ATH window is mostly noise** (17 possessions). P4 variants moved between 0.60 and 0.71 there with
   no consistent pattern. Never choose on it (RULES.md rule 2).
7. **Validation becomes optimistic with reuse:** after about 10 decisions, P4 dropped 0.017 from validation to test.
   Treat validation leads under about 0.02 as unproven.
8. **Pooling seeds in `keep_rule` reuses the same chunks,** so the interval is somewhat too narrow. Be careful with
   changes that only just pass.

## Where the errors are

9. **Missing-ball frames are the weak spot:** live-play accuracy about 0.81 there against about 0.885 with the ball
   visible. Report the visible / missing split (`visibility_report` in `scripts/train_p4.py`).
10. **The model alone breaks possession into far too many spells,** and the smoother does most of the work: on test
    the unsmoothed output has 832 spells against 294 true. Unsmoothed spellF1 0.36-0.44, smoothed 0.69-0.71. Tune
    the penalty on your own probabilities; P4 landed on 10-20.

## Engineering

11. **Determinism (rule 3):** `torch.manual_seed`, `torch.use_deterministic_algorithms(True)`, CPU, pinned
    `torch==2.14.1` gave byte-identical reruns of training and of the ensemble. Check this early.
12. **Put every setting in the run name.** P4 nearly overwrote its best run when hidden size was not in the name.
13. **With 4 training matches a neural model overfits fast:** early stopping stopped at epoch 9-13. A bigger model is
    unlikely to help without more data.
