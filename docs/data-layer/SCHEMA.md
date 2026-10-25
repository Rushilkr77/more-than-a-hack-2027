# Frame table schema (v0)

GENERATED from `SCHEMA` in `src/features/frame_table.py` by `scripts/render_schema_doc.py`. Do not edit by hand; a test fails if this drifts.

One row per tactical-video frame (199,319 rows), `frame` 0-based, row i == frame i. Load it with `frame_table.load()`.

Conventions
- Raw position columns (`*_cx_m`, `ball_x_m`, ...) are metres on a 105.3 x 68.0 m pitch in the ATD's own orientation (NOT flipped per half).
- Shape and ball-territory columns (`*_dline_m`, `*_cen_m`, `*_fwd_m`, `*_line_gap_m`, `ball_home_adv_m`) are "metres from that team's own goal line": bigger = more advanced, same meaning in both halves, via `home_attacks_right`.
- Missing is NaN for floats; counts are 0; booleans are never null. A missing ball is `ball_visible == False`, never "dead ball".
- Reliability columns: `home_shape_conf`, `away_shape_conf` (0..1), `ball_margin_m` (bigger = more trustworthy `nearest_team`). Hints (`tag_poss_hint`) are noisy weak labels.

| column | dtype | meaning |
|---|---|---|
| `frame` | int32 | tactical video frame, 0-based; `frame_id - 1` in the ATD. Row i has frame == i |
| `period` | int8 | 1 = frame <= end of first graded block (includes pre-match), 2 = frame >= start of second graded block (includes post-match), 0 = halftime gap |
| `graded` | bool | frame is inside a graded block (158,857 frames in total) |
| `pre_kickoff` | bool | graded but before that half's kickoff (second half: 110808..113741; evaluator labels these D) |
| `ball_visible` | bool | ball has both coordinates in the ATD. Visibility is evidence quality, NOT dead-ball state |
| `ball_x_m` | float32 | ball x, metres, NaN if not visible |
| `ball_y_m` | float32 | ball y, metres, NaN if not visible |
| `ball_speed_atd_ms` | float32 | ATD ball speed field as delivered: ~91% NaN even when the ball is visible, unreliable, do not use |
| `n_home` | int8 | home tracks visible (slots are tracklets, not players; can exceed 11) |
| `n_away` | int8 | away tracks visible |
| `home_cx_m` | float32 | home: mean x of visible tracks, metres |
| `home_cy_m` | float32 | home: mean y of visible tracks, metres |
| `home_depth_m` | float32 | home: std of x (length-wise spread), metres |
| `home_width_m` | float32 | home: std of y (width-wise spread), metres |
| `home_speed_ms` | float32 | home: mean speed of visible tracks, m/s |
| `away_cx_m` | float32 | away: mean x of visible tracks, metres |
| `away_cy_m` | float32 | away: mean y of visible tracks, metres |
| `away_depth_m` | float32 | away: std of x (length-wise spread), metres |
| `away_width_m` | float32 | away: std of y (width-wise spread), metres |
| `away_speed_ms` | float32 | away: mean speed of visible tracks, m/s |
| `home_ball_dist_m` | float32 | distance from ball to nearest visible home track, metres; NaN if ball or tracks missing |
| `away_ball_dist_m` | float32 | distance from ball to nearest visible away track, metres; NaN if ball or tracks missing |
| `nearest_team` | int8 | team of the track nearest the ball: 0 home, 1 away, -1 unknown (the official baseline's signal) |
| `hom_valid` | bool | tactical homography matrix exists for this frame |
| `hom_reproj_err` | float32 | homography reprojection error (lower is better); ~1.5 across halftime |
| `tag_n_active` | int16 | Smart Tagging intervals covering this frame (noisy evidence, not possession) |
| `tag_home_n` | int16 | active intervals tagged to the home team |
| `tag_away_n` | int16 | active intervals tagged to the away team (N/A-team intervals count for neither) |
| `tag_setpiece` | int8 | active SET PIECES type: 0 none, 1 throw-in, 2 corner, 3 goal kick, 4 free kick, 5 kick-off (start), 6 kick-off (after goal), 7 other |
| `tag_setpiece_team` | int8 | team of the active set piece: 0 home, 1 away, -1 none or N/A (every throw-in) |
| `home_attacks_right` | bool | home attacks toward +x this half (ATD XML attack_direction_first_half, flipped in the second half; halftime gap uses the first-half value) |
| `ball_margin_m` | float32 | |nearest home dist - nearest away dist| to the ball; confidence for nearest_team (two cameras agree on the nearest team 67% of the time below 0.5 m, 99% above 8 m); NaN if either is missing |
| `ball_home_adv_m` | float32 | ball distance from the home goal line along the home attack axis; away = pitch length minus this; NaN if ball missing |
| `ball_third_home` | int8 | ball third from the home point of view: 0 home defensive, 1 middle, 2 home attacking; -1 unknown |
| `home_dline_m` | float32 | home: defensive line: mean metres-from-own-goal of the 2nd-5th deepest on-pitch tracks (GK excluded) |
| `home_cen_m` | float32 | home: outfield centroid: mean metres-from-own-goal of all on-pitch tracks except the deepest |
| `home_fwd_m` | float32 | home: forward line: mean metres-from-own-goal of the 3 most advanced tracks after trimming the single most advanced |
| `home_line_gap_m` | float32 | home: fwd_m - dline_m: how stretched the team is between its lines |
| `away_dline_m` | float32 | away: defensive line: mean metres-from-own-goal of the 2nd-5th deepest on-pitch tracks (GK excluded) |
| `away_cen_m` | float32 | away: outfield centroid: mean metres-from-own-goal of all on-pitch tracks except the deepest |
| `away_fwd_m` | float32 | away: forward line: mean metres-from-own-goal of the 3 most advanced tracks after trimming the single most advanced |
| `away_line_gap_m` | float32 | away: fwd_m - dline_m: how stretched the team is between its lines |
| `home_shape_conf` | float32 | reliability 0..1 of the home shape columns: clip((on-pitch tracks - 5) / 5, 0, 1); 0 if fewer than 7 tracks (columns NaN) |
| `away_shape_conf` | float32 | reliability 0..1 of the away shape columns (same rule) |
| `tag_poss_hint` | int8 | weak possession hint from Smart Tagging phases: +1 home, -1 away, 0 none or conflicting (automatic tags, noisy) |
