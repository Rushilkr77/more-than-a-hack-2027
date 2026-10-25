# Broadcast tracking: what it is and why fills are optional

The broadcast recording has its own ATD (164,488 frames, 51 slots) and homography (65.8% valid). It is on a different timeline: cuts, replays, no mapping provided.

## Alignment (found, not given)
Both ATDs use the same 0-1 pitch coordinates, so player configurations can be matched directly. The broadcast is the tactical timeline shifted by a constant, one per half:
`tactical = broadcast + 4014` (first half) and `+ 26920` (second half); halftime is cut out. Evidence: player-configuration cost has a clear minimum at those offsets (0.703 and 0.679 m mean nearest-neighbour distance vs ~8.6 m for a wrong shift), and the ball independently agrees (median 0.8 m, 74% within 2 m on aligned frames vs 0.2% for a random shift). Offsets re-fit to the same values (`broadcast_xcheck.py --refit`); +/-1 frame changes tier counts by <3%. The split between the two offsets is hard-coded at tactical frame 100000 (the halftime gap is 90652..110807).
Replays and cuts appear as frames whose players do not match at the mapped time.

## Confidence tiers (per tactical frame, pooled over +/-125 frames)
| tier | meaning | conf | share of graded frames |
|---|---|---|---|
| 1 verified | alignment verified and ball agreement observed and good | 0.75-1.00 | 46.7% |
| 2 partial | mostly aligned, or ball agreement unobserved | 0.50-0.75 | 25.9% |
| 3 mismatch | large mismatch; filling is a risk | 0.25-0.50 | 7.1% |
| 4 unverifiable | no evidence either way | NaN | 20.4% |
Tiers are ordinal grades from measured agreement, not calibrated probabilities. Thresholds were set by eye (see RISKS 13).

## Confidence matrix (graded frames)
| tier | gap, broadcast ball available | gap, no broadcast ball | tactical ball present |
|---|---:|---:|---:|
| 1 | 3,936 | 9,042 | 61,164 |
| 2 | 2,518 | 17,804 | 20,754 |
| 3 | 1,799 | 3,699 | 5,742 |
| 4 | 0 | 25,845 | 6,554 |
Of 64,643 graded gaps, the broadcast has a ball for 8,253 (13%); tiers 1-2 would fill 6,454 (10%). Most gaps are frames where both cameras lost the ball.

## How good are fills?
Where both balls are visible, the broadcast ball picks the same nearest team as the tactical ball 91.5% (tier 1), 74.2% (tier 2), 69.4% (tier 3). In the burst-mask test the fill is right 89.7% / 83.6% (tiers 1/2). Fills raise frame-level accuracy and macroF1 but do not improve spellF1 (no variant; best -0.003). Hence: **optional, off by default, never in the baseline build.**

## Using fills
```bash
.venv/bin/python scripts/broadcast_xcheck.py                              # builds/refreshes the cross-check cache, writes outputs/broadcast_fill_segments.csv
.venv/bin/python scripts/build_submission.py --config configs/filled_t12.json   # prints a WARNING; manifest says uses_broadcast
```
Evidence columns: `ball_source` (0 tactical, 1/2 fill tier, -1 none), `weight` (1.0 tactical; fill = alignment conf), `anchor` (1 tactical, 0 fill), `align_conf`. Marked stretches: `outputs/broadcast_fill_segments.csv` (start, end, tier, mean weight, graded).

## Caveats
Fill accuracy was measured on easier frames than the real gaps; filled frames vote 48.2% home vs 56.4% on tactical-ball frames (cause unknown); the broadcast file path may be nested (code searches one level down). Another match needs its own offsets.
