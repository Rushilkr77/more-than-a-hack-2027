# Measured findings

Each line was checked on the provided match with a script in `scripts/` (most by `audit_data.py`). Numbers are for graded frames unless stated.
Confidence: **verified** = independent check; **measured** = computed, no independent check; **inferred** = reasoned.

## Loader and timeline
- 400 random frames match an independent plain-python re-parse of the raw file: 0 mismatches. **verified**
- `frame = frame_id - 1`; the `games.py` per-half offsets do NOT apply to the delivered ATD. 199,319 rows; 158,857 graded (79,467 + 79,390); 2,934 pre-kickoff frames at the start of the second graded block. **verified**
- Our baseline CSV equals the published `baseline_smooth50.csv` byte for byte (sha256 6d4e04b8...). **verified**
- Throw-in starts sit at a touchline for 86% of 14 throw-ins with a ball: time base and orientation are consistent. **measured (small n)**

## Orientation
- ATD XML: Barcelona (home) attacks left to right in the first half, Atletic right to left; sides swap in the second half. Measured mean team positions agree (H1 home x 48.6 vs away 61.4; H2 55.8 vs 43.9). **verified**
- Raw position columns are NOT flipped. Use `home_attacks_right` or the own-goal-distance columns.

## Ball
- Visible on 56.9% of all frames, 59.3% of graded. Missing runs: 150 graded runs, median 10.7 s, p90 33.8 s, longest 221 s; 85% of missing frames lie in runs over 10 s. **measured**
- Missingness is not random: P(ball missing 1 s later) is 8.0% after being seen in a penalty box vs 2.5% in midfield; the ball is missing at the end of every GOALS phase. **measured**
- 2.07% of visible graded ball observations are more than 2 m outside the pitch (379 are >10 m out); only 24% of them fall near restarts (chance 5.8%), so out-of-bounds is not a reliable dead-ball cue. **measured**
- `ball_speed_atd_ms` is blank on 91.3% of ball-visible frames and does not track position-derived speed where present. The XML labels the speed sensor "kph" but player values verify as m/s (ratio 0.99, corr 0.96). **verified**
- Position jumps imply >40 m/s on 2,043 of 113,303 consecutive pairs: ball tracker teleports exist. **measured**

## The nearest-player signal (baseline)
- Home share of `nearest_team` is 56.41% (56.44% with on-pitch tracks only; flips 184 vs 209). **measured**
- Nearest player is >5 m from the ball on 19% of visible graded frames; the margin between teams is <1 m on 17% (contested). Two cameras agree on the nearest team 67% (margin <0.5 m) up to 99% (margin >8 m): `ball_margin_m` is a real confidence. **verified (vs broadcast)**
- It leans toward defenders: "ball in home attacking half" predicts home nearest-team only 40.8%. It is a proxy, not possession.
- Switches 19 per 1,000 ball-visible frames, 26 inside LONG BALL phases.

## Tracks
- 46 slots (home 24, away 22); slots are tracklets, not people. Mean visible tracks 20.5 per graded frame. Positions only; no roles (44 of 46 slots `position_type` Unknown, two goalkeepers flagged). **verified**
- More than 11 tracks: home 8.1%, away 4.7% of graded frames (max 24). Goalkeeper area empty: home 15.8%, away 21.7%. Excess tracks are on the pitch, so a pitch mask does not remove them. **measured**
- Halftime is not empty: mean 12.5 visible tracks (the starter docs say "almost none"). **measured**

## Team shape (own-goal distance, metres)
Tagged team's mean defensive line / centroid during Smart Tagging phases: defending in defensive third 16/23; build up 28/41; attacking transition 35/46; progression 51/61; players in the box 59/71; ball in the box 64/75; creating chances 66/76. Shape tracks phase of play without the ball. **measured against noisy automatic tags**
- Shape does NOT predict the nearest-player proxy (48.7%): validate against phases or other-match labels, not against `nearest_team`.
- Usable on ~95% of graded frames (>=7 on-pitch tracks). Typical style: home defensive line median 44-45 m, away 38-40 m.
- No formation information exists; kickoff snapshots have 10-11 tracks and sometimes a missing goalkeeper track. Formation labels would be unreliable.

## Smart Tagging
638 automatic phases; `Team` is N/A on 36 including every throw-in; boundaries approximate. `tag_poss_hint` is non-zero on 57% of graded frames; its accuracy is unmeasured.

## Baseline behaviour (labels, not truth)
Graded labels: H 38.6%, A 29.1%, D 32.3%; 594 spells (the public five minutes have 17 over 5 minutes). Home share of alive time 57%, matching the public window (57.3%).
