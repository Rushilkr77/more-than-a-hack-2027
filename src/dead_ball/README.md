# Dead-ball probability (draft)

Per-frame probability that the ball is dead (`p_dead`, 0 to 1) for the BAR-ATH tactical match. Standard library only.

> **Draft, not for submission as it stands.** The competition rules ban hand labels on the provided game
> ([RULES.md](../../RULES.md)). Two parts of this module use them or constants read off the game and must be replaced
> before it feeds a submission:
> 1. the final calibration step (`cal_*` in `CFG`), fitted to hand-labelled stoppages;
> 2. the hard-coded play periods and half-time (`play_periods`, `halftime` in `CFG`), read from the video.
>
> The evidence score `E` below uses no fitted values, only hand-chosen weights and thresholds.

## Run

```bash
python src/dead_ball/dead_ball.py            # reads data/tactical/ (or $MTAH_DATA_DIR/tactical/)
python src/dead_ball/dead_ball.py --eval     # also prints a tag-based sanity check
```

Writes `outputs/dead_ball/frame_scores.csv` (one row per frame) and `dead_segments.csv` (one row per stoppage).
`outputs/` is gitignored. Frames outside play are left blank and labelled `HALFTIME` or `OUT_OF_PERIOD`.

## Method

Signals, each scaled 0 to 1 by `ramp(v, a, b)` (0 at `a`, 1 at `b`, linear between):

| signal | from | scaling |
|---|---|---|
| `s_tag` | `SET PIECES` tag in the Smart Tagging XML | 1 from 1 s before to 3 s after the tag start, ramps in and out |
| `s_still` | ball movement range `r` over ±0.5 s (ball visible) | `ramp(r, 3 m, 1 m)` |
| `s_out` | ball distance `d` beyond the pitch line | `ramp(d, 0.3 m, 1.5 m)`, only if `r` < 3 m |
| `s_pl` | outfield player speeds over ±1 s | `0.5·ramp(slow, .20, .55) + 0.5·ramp(fast, .12, .02)` |
| `live` | ball range `r` | `ramp(r, 3 m, 8 m)` |

`slow` and `fast` are the shares of players under 1 m/s and over 3 m/s. A missing ball adds no evidence.

```
data = 1 - (1 - .60*s_still)(1 - .50*s_out)(1 - .60*s_pl);  data *= 1 - .75*live
E    = 1 - (1 - .90*s_tag)(1 - data)
m5   = mean of E over ±5 s (in-play frames)
z    = -1.465 + 0.519*(E - 0.254)/0.311 + 0.933*(m5 - 0.253)/0.234
p_dead = 1 / (1 + exp(-z))
```

Dead-ball segments, `reason_code` and `confidence` use `E` (threshold 0.5, at least 2 s), not `p_dead`.

## Known limits

- Weights and thresholds are hand-chosen, not learned.
- The ball is invisible on about 40% of frames; then only player speeds and tags are available.
- Tags are automatic and timed only to a few seconds.
- Long stoppages with players standing in groups (drinks break, substitution) are only partly detected.
- The restarting-team guess (marked `?`) is right about 60% of the time when there is no tag.
