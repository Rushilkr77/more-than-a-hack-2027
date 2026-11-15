# What P2 taught us (for P4 and the other approaches)

Measured on `main` at `b67351c` (common input contract merged). Comparisons are only valid with the same split,
smoother and scorer as `docs/common/COMMON_LAYER.md`.

## Numbers to beat

spellF1 against `labels_adjusted` (Sportec) and the public BAR-ATH window. P2 = `h_sd_win_1_5_15`, no `gtag_*`, mean of 3 seeds.

| | Sportec validation (2 matches) | BAR-ATH public 5 min (17 possessions) |
|---|---|---|
| P2 smoothed (penalty 20) | 0.671 | 0.734 (interval about 0.53-0.90) |
| P2 unsmoothed | 0.313 | 0.288 |
| Organiser baseline smooth50 | 0.219 | 0.326 |
| Organiser baseline raw | 0.049 | 0.039 |

Final-submission form is the 3-seed probability ensemble (`python scripts/train_p2.py --ensemble`): Sportec test
0.662 smoothed / 0.345 unsmoothed, BAR-ATH 0.735 / 0.380. Full table in `docs/models/P2.md`. If P4 submits an ensemble,
compare it to these; during development compare single-seed means.

Report both smoothed and unsmoothed. P2's frame-level macroF1 is about 0.90 on Sportec and 0.84 on BAR-ATH.
Validation was reused for about 30 tuning decisions, so 0.67 is optimistic. The Sportec test match J03WQQ has been scored for P2 (single seeds 0.653 smoothed / 0.300 unsmoothed); do not tune on it.

## Protocol P2 uses (copy it so numbers are comparable)

- Train on `labels_raw`, every 5th frame, plus the home/away `sportec.mirrored` copy of each training match.
- Early-stop on the 2 validation matches; tune the shared Viterbi penalty (grid `PENALTIES`) on validation `labels_adjusted`.
  Never tune on BAR-ATH. P2 picks penalty 20 (10 and 40 are close, so the choice is not fragile).
- Inputs only via `sportec.classification_inputs(match, InputPolicy(schema, mode))`. P2 uses `04_full_phase_context`,
  `OFFLINE_ACAUSAL`. P4 can take the structured arrays (`players`, `ball`) from the same call.
- Windows: P2 adds past/next 1, 5 and 15 s mean, std and diff within each half. A GRU sees the sequence directly, so it
  should not need these.
- BAR-ATH check: `frame_table.load()` and `load_slots()` give the table and slots; pitch comes from
  `data/tactical/BAR-ATH_tactical_atd.xml`; score against `evaluation/gt/BAR-ATH_possession_gt_public.csv` with
  `sportec.score` and `possession_metrics.bootstrap_spell_f1` (1000 draws). Only 17 possessions, so gaps under about
  0.1 are not significant. Use several seeds.

## What we tried and what to reuse

- **`gtag_*` (generated tags): leave out.** The whole block lifts Sportec by about 0.03 but costs about 0.09 on BAR-ATH.
  `gtag_poss` is the cause (nearest-team proxy). The other seven tags add at most 0.006 on Sportec (seed noise) and are
  flat or lower on BAR-ATH. This is DECISION_LOG OPEN-06's cross-dataset rule: a Sportec-only gain does not count.
- **Smoothing is essential.** Unsmoothed P2 predicts about 3x too many spells (Sportec) and about 3-4x on BAR-ATH.
  Use the common smoother for the headline number; an own smoother is fine as an extra row.
- **Seed noise:** about 0.005-0.013 on Sportec and 0.002-0.05 on BAR-ATH. Do not trust single-run differences.
- **Dead-ball is the weak spot** (earlier finding: about half of dead-ball frames recovered on the 5-minute window,
  from 2 stoppages). Dead ball is about 44% of Sportec frames but about 20% of the public clip.

## Practical gotchas

- `data/` and `cache/` are not in git. In a fresh worktree, remove the tracked `data/` stub first, then symlink the real one
  (`rm -rf data && ln -s <repo>/data data`), or `load` will not find the bundles.
- Copy `cache/` into a worktree rather than symlinking it; a stale cache rebuilds and would overwrite the shared one.
- `sklearn>=1.7` is only needed for P2; a GRU needs its own dependency in `requirements-team.txt`.
- Do not use the test match for any decision.
