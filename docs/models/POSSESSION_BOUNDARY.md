# Learned possession boundaries

## Decision log

1. **Choose the decoder as the intervention.** The incumbent in PR #16 reports
   0.462 model-only and 0.727 smoothed Sportec validation spell-F1. Its fixed
   transition-type costs and minimum durations tied. A tracking-dependent cost
   is a different hypothesis. Keep the frozen member probabilities identical
   between control and candidate.
2. **Freeze v1 before scoring.** Use the shared offline schema 03 with 15
   team-symmetric tracking features, ±0.4 s same-period context, native
   missing-value trees with 120 iterations, and one sample every five frames.
   Fit on the four whole training matches with raw edge targets within ±0.2 s
   of an H/A/D change. Unknown labels and period starts cannot create edges.
   Swapping H/A leaves the target and features invariant, so duplicating the
   mirrored samples adds no information.
3. **Use one decoder setting.** The published incumbent cost of 20 stays
   fixed for both decoders. The candidate subtracts the boundary log-odds
   uplift over its training prior from that cost, bounded to a ±10-nat
   correction. The model cannot make a switch free in this experiment.
4. **Use the existing promotion gate.** Compare on the two whole validation
   matches using adjusted labels, the common scorer, and its paired two-minute
   bootstrap. Require a strictly positive interval and non-negative whole-match
   spell-F1 gain on each match. A short check with available P2 and GRU-30
   artifacts is reported separately from the full three-member recipe. The
   gate primarily concerns the full recipe. No held-out or BAR-ATH labels
   enter the run.

The boundary-prediction motivation comes from
[Ishikawa et al.](https://arxiv.org/abs/2007.06866). This implementation tests
input-dependent transition costs on soccer tracking, with a separate tree
detector and the repository’s shared Viterbi implementation.

## Reproduce

Generate the three member directories as described in [BLEND.md](BLEND.md).
The boundary experiment needs only the validation probabilities, so the
member commands can omit `--export`. Build or link the standard processed
Sportec bundles and labels first.

```sh
python scripts/evaluate_possession_boundary.py --members \
  outputs/p2/h_sd_win_1_5_15_ens \
  outputs/p4/04_g0_w30_s5_aug2_ens \
  outputs/p4/04_g0_w20_s5_aug2_ens
```

The ignored output directory holds per-match label CSVs, `events.jsonl`, and
`meta.json` with the source revision, library versions, hashes of the input
arrays, match scores and keep result. Choose a new directory with `--output`
for each run.

## Validation log, 2026-11-17

Run revision `2753216`, clean tree, Python 3.12.12, scikit-learn 1.7.2,
NumPy 2.5.5. The P2 member was regenerated with explicit whole-match validation.
The GRU-30 member was regenerated with PyTorch 2.8.0. These are locally
regenerated artifacts, with different scores and provenance from PR #16's
published artifacts.

| Two-member check | Fixed spell-F1 | Learned spell-F1 | Fixed macro-F1 | Learned macro-F1 | Fixed / learned / true spells |
|---|---:|---:|---:|---:|---:|
| J03WMX | 0.67714 | 0.68327 | 0.91191 | 0.91236 | 246 / 248 / 314 |
| J03WOY | 0.72313 | 0.72068 | 0.90325 | 0.90460 | 273 / 275 / 315 |
| Mean | 0.70014 | 0.70198 | 0.90758 | 0.90848 | |

Paired mean chunk gain **+0.00364**, 95% interval **[-0.00266, +0.01095]**,
97 chunks. The check fails both the interval and the whole-match direction
requirements. Boundary average precision was 0.3720 / 0.3802 at a positive share
of 0.0313 / 0.0328. The detector learned a signal, but this check did not show
an actionable improvement in decoded spells.

```text
J03WN1 rows=28312 positives=1298
J03WOH rows=27442 positives=1083
J03WPY rows=29242 positives=1103
J03WR9 rows=29361 positives=990
train_prevalence=0.03912310
output=outputs/possession_boundary/v1_p2_gru30
keep=false
```

The three-member comparison is pending regeneration of the GRU-20 artifacts.
