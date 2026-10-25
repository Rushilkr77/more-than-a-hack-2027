# Risks

Severity = how much it can hurt spellF1/macroF1 or the submission's validity. "Status" says whether it is mitigated in code, only documented, or open.

| # | risk | severity | evidence | mitigation / what to do | status |
|---|---|---|---|---|---|
| 1 | Nearest-player is a proxy for possession and leans toward defenders; a model trained to imitate it learns the proxy's errors | high | ball-in-home-attacking-half predicts home nearest only 40.8%; nearest >5 m from ball on 19% of frames | use `ball_margin_m` as confidence; never validate shape features against `nearest_team`; judge on other-match labels | documented |
| 2 | Ball missing 41% of graded frames, in long runs (85% of missing frames in runs >10 s), more often near the box | high | FINDINGS.md | ball-free evidence (shape, dead-ball) must carry long gaps; interpolation will not help | open (modules to build) |
| 3 | Broadcast fills do not improve spellF1 | medium | burst-mask test, 8 seeds: no variant improves it; best (tiers 1+2, weights only) -0.003, 4/8 seeds up | fills are optional, off by default, configs marked `uses_broadcast`; re-test as a model input before use | mitigated |
| 4 | Fills pick the wrong team 10% (tier 1) / 16% (tier 2) of the time; wrong fills create short wrong spells | medium | same test | weight and `anchor`/`min_anchor`; but `min_anchor=1` lowered spellF1 by ~0.04 | documented |
| 5 | Fill validation is optimistic: measured where both balls are visible, reference is the unmasked tactical decode (not truth), random bursts are easier than real gaps, and real gaps have a broadcast ball only ~13% of the time | medium | validate_fill.py header | read coverage and per-fill accuracy, not only net score; expect real effect ~5x smaller than in the test | documented |
| 6 | Polluted tracks: >11 tracks per team (home 8.1%, away 4.7%), ghost/duplicate tracks, goalkeeper area empty (home 15.8%, away 21.7%) | medium | audit_data.py | shape columns use trimmed statistics and `*_shape_conf`; never hard-code slot ids (rule: no constants tied to this match) | mitigated for shape; open for new features |
| 7 | Home/away asymmetry: away data is noisier; 57% home share is a real prior of this match that a model can memorise | medium | counts above | team-relative features, no team names, `swap_sides()` augmentation, antisymmetry tests, per-side recall in reports, neutral `tie="previous"` for new configs | mitigated |
| 8 | Baseline tie rule favours home (175 vote ties in 158,857 graded frames) | low | bias audit | kept only so the baseline stays byte-identical; new configs use `previous` | mitigated |
| 9 | Orientation: raw positions are not flipped between halves; features that ignore it mix attack and defence | high if ignored | home mean x 48.6 vs 61.4 (H1), reversed in H2 | use `home_attacks_right` or the own-goal-distance columns | documented |
| 10 | `ball_speed_atd_ms` is 91% blank and unreliable; XML says kph, players verify as m/s | medium | FINDINGS.md | do not use; derive speed from positions with jump filtering only if a feature needs it | documented |
| 11 | Out-of-bounds ball is not a reliable dead-ball cue; 2% of ball positions are outside the pitch, some by >10 m | medium | 24% near restarts vs 5.8% chance | do not treat as D; treat >10 m outliers as junk | documented |
| 12 | `tag_poss_hint` accuracy unmeasured; Smart Tagging is automatic, Team N/A on every throw-in | medium | DATA.md, FINDINGS.md | treat as a weak label with agreement check; measure on other matches before trusting | open |
| 13 | Broadcast alignment constants (offsets 4014 / 26920, split at tactical frame 100000) were fitted on this match; thresholds (3 m, 5 m, 0.9/0.6/0.8, window 125) were set by eye | medium | offsets stable to +/-1 frame; window changes tier mix a lot (tier 1: 984 candidates at 25 frames, 4,210 at 250) | re-run `broadcast_xcheck.py --refit`; any other match needs its own offsets | open |
| 14 | No ground truth for this match; the public window has 17 possessions and picks the wrong smoothing window (see EVALUATION.md) | high | repo docs | validate on other matches with labels, burst-mask tests, ablations; never tune on the public five minutes | open |
| 15 | Other-match validation data is not yet chosen; format/noise may differ (cleaner tracking flatters features) | high | no compatible dataset confirmed | check labels, tracking noise and licence first; add synthetic corruption | open |
| 16 | Reproducibility: caches need the raw bundle to rebuild; Python 3.14 environment; one-command rebuild must work on a clean clone | medium | fingerprint design | `requirements-team.txt`; clean-machine rehearsal before submission | open |
| 17 | Licence: the frame table, outputs, manifests and segment files are derived frame-level data | high | RULES.md "do not publish derived data that would reconstruct it" | `cache/`, `outputs/`, `data/` are gitignored; never commit them or paste raw frames into hosted tools | mitigated |
| 18 | Metric blind spot: slicing cannot attribute spell errors; spell metrics are global | low | slice_report.py | use manifests plus slice report together; compare runs by manifest diff | documented |

## Before you rely on a column, ask
1. Is it a measurement, an inference, or a weak label? (SCHEMA.md and FINDINGS.md say which.)
2. What is its reliability column, and what happens when that is low?
3. Does it behave the same for home and away?
4. Was it validated against anything other than the proxy it correlates with?
