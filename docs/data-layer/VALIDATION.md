# Validation and debugging

## What is allowed (RULES.md, EVALUATION.md)
- No hand labels or corrections on the provided match, for any purpose. No private ground truth.
- The public five minutes (frames 11185..18684, 17 possessions) are for checking alignment and file format only. Not training data, not a tuning set: they rank the smoothing windows wrongly.
- Labels from OTHER matches are fine in any amount if organisers can access them and the technical document explains source, method and amount. Split by whole matches, never random frames.
- Algorithmic pseudo-labels on the provided match are fine (no person decides them).

## Proxies available now (none is truth)
| tool | what it tells you | caveat |
|---|---|---|
| `scripts/validate_fill.py` | hides the tactical ball in 0.5-10 s bursts, re-grades, measures recovery of the unmasked decode | reference is the unmasked baseline, not possession; random bursts are easier than real gaps |
| `scripts/audit_data.py` | facts about the data, shape-vs-phase check | phases are automatic tags |
| burst-mask / corruption tests | does a module survive missing ball, dropped tracks, jitter | needs writing per module |
| ablation | remove one input at a time | needs a labelled validation match to mean anything |

## Finding what hurt spellF1/macroF1
1. Diff the two runs' `*.manifest.json` (config, code hash, evidence shares, label mix).
2. Run `scripts/slice_report.py --pred new.csv --gt labels.csv` for a validation match, then the same for the old prediction. It slices by half, ball visible/missing, ball margin, shape confidence, weak possession hint, with H/A/D recall and each slice's share of all errors. A slice whose error share jumped is the suspect.
3. Check H and A recall separately and `share_err` (home share error): a one-sided drop means bias, not noise.
4. Spell metrics are global; if spellF1 fell but macroF1 did not, look for fragmentation (more, shorter spells) in the manifest label mix and the spell counts.
Use `--proxy other.csv` to see where two predictions of the same match differ; that is a difference report, not accuracy.

## Other-match data (planned, not chosen)
Allowed in principle. Check before committing: (1) does it have possession labels, or events we would have to turn into labels; (2) does the tracking have the same defects (missing ball, duplicates), otherwise add synthetic corruption so reliability features are tested; (3) licence and accessibility for the organisers; (4) conversion to the 0-1 normalised, ball-plus-slots format. Offsets and thresholds from the broadcast cross-check do not carry over.

## Selection discipline
Promote a model only if it improves spellF1 (then macroF1) on whole held-out matches and under ball-masking, has an ablation-backed reason, and regenerates from raw inputs with one command. Record every run's manifest.
