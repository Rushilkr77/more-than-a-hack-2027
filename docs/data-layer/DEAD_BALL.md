# Dead-ball columns

`D` means the ball is not in play (CHALLENGE.md). That includes the wait before a corner, throw-in, goal kick, free kick or kick-off. It ends when the ball is played. These columns are **model inputs, not labels**. P2–P5 learn how to combine them. Nothing is fitted on the provided match.

Built by `dead_ball_columns()` in `src/features/frame_table.py`; meanings and formulas are in [SCHEMA.md](SCHEMA.md) (generated, the contract).

## Why these columns
| case | cue in tracking | columns |
|---|---|---|
| any stoppage (injury, substitution, celebration) | players slow or stopped | `pl_slow_share`, `pl_fast_share` |
| restart boundary | ball still, then kicked | `ball_range_back_1s_m`, `ball_range_fwd_1s_m`, `ball_speed_p90_ms` |
| throw-in | ball on or beyond a touchline; a still track on the line | `ball_dist_touchline_m`, `pl_still_at_line_n` |
| corner | ball near a corner flag; crowd in one box | `ball_dist_corner_m`, `ball_dist_goalline_m`, `pl_in_box_max_n` |
| goal kick | ball near a goal, inside the box | `ball_dist_goal_m`, `ball_dist_goalline_m` |
| kick-off | ball on the centre spot; each team in its own half | `ball_dist_centre_m`, `home_own_half_share`, `away_own_half_share` |
| free kick | still ball, players gathered | ball range + Kalhan's near-ball support counts (no unique signature: expect it to be the weakest case) |

## Rules
- **A missing ball is no evidence.** Ball columns are NaN and `dead_ball_conf` is 0. The ball-free columns (`pl_*`, own-half shares) still carry the frame, which matters because the ball goes missing more often near the boxes.
- **A still ball alone is not dead.** A keeper holding the ball or a player shielding it keeps it in play. The models learn the combinations.
- **Constants** are rule-book geometry or physical units: penalty area 16.5 × 40.32 m, 1.5 m from the line, 1 and 3 m/s, 40 m/s jump, 10 m junk. None was tuned on this match or read off the video.
- **Windows** never cross a `period` change. Pre-kickoff and halftime frames are computed like any others.
- **No goalkeeper flag.** Slots carry no reliable role, so the deepest on-pitch track per team is dropped from the speed shares (same trim as the shape columns).
- **Track counts are inflated** by tracklet duplicates on ~8% / 5% of frames.
- **Smart Tagging is not used here.** It belongs to the smoother only.

## How it flows ahead
`dead_ball_columns()` is a pure function over arrays (`x, y, s, team, ball_x, ball_y, period, home_attacks_right`, pitch size, fps). `build_table()` calls it for this match; the input-layer methods call the same function for any other match in the same array format. The columns go to the models as inputs; how they are prepared and labelled for training is documented with the input layer, not here.
