# Standalone AI input-file schemas

## Why these are standalone

Each CSV has all fields needed for its own experiment:

- frame and half information;
- shape-quality information;
- Home and Away team fields;
- missing-data flags where needed.

A model can read one CSV without joining another CSV.

## Why we still keep one master feature table

One master feature table is the correct storage layer for architecture-agnostic model work. It contains the union of every approved feature once.

The four CSVs in this folder are exports from that master table. They are controlled experiments, not separate sources of truth.

This lets us answer useful questions:

1. Does basic shape work at all?
2. Do time-based shape changes add value?
3. Does ball-and-gap information add value?
4. Does automatic phase context add value?

## Input files

| File | Standalone contents | Main experiment |
|---|---|---|
| 01_core_shape.csv | Basic team shape, speed and quality | Ball-free baseline |
| 02_shape_dynamics.csv | Core shape plus short-window movement and line coordination | Does movement add value? |
| 03_ball_shape_extended.csv | Core shape plus ball, ball-gap and extended ball-relative spatial fields | Does ball context add value? |
| 04_full_phase_context.csv | Core shape, dynamics, ball context and direct Smart Tagging context | Does available phase context add value? |

## Important rules

- Empty numeric values mean missing. Do not replace them with zero.
- Every input has the quality columns so models can learn when not to trust a measurement.
- Do not use nearest_team as a training target or a possession label.
- File 04 uses direct Smart Tagging only when that feed is available when the final model will run.
- Phase probabilities from a separate phase model are a later replacement experiment for direct tags, not a column copied from the current tag label.

## Shared fields: why every schema has them

| Field group | Fields | Why every input needs them | Decision context |
|---|---|---|---|
| Record identity | frame_id, period, home_attacks_right | Keeps records in video order and makes forward/backward comparable after teams switch ends at half-time. | Without period and attacking direction, a model can confuse a forward movement in the first half with the same raw x movement in the second half. |
| Track availability | n_home, n_away | Says how many Home/Away tracks are visible. | A team-shape number is less trustworthy when the tracker sees too few players or duplicates tracks. |
| Shape quality | home_shape_conf, away_shape_conf, home_shape_missing, away_shape_missing | Gives the model an explicit quality score and missing-value indicator. | Missing data must not be changed to zero. A model needs to distinguish “team is narrow” from “team width is unknown.” |

The repository defines shape confidence as a value from 0 to 1 and leaves shape values unavailable when too few usable tracks exist. See the [frame-table schema](../../data-layer/SCHEMA.md).

## Schema 1: 01_core_shape.csv

| Item | Decision |
|---|---|
| Input type | Standalone frame-level team-shape baseline. One row describes one tactical-video frame. It can also be read as a time sequence by a temporal model. |
| Fields | Home and Away dline_m, cen_m, fwd_m, width_m, depth_m, speed_ms, plus all shared fields. |
| Why this input type | It answers the first useful question: can team arrangement alone provide useful information when no ball fields or automatic tags are supplied? |
| Research cited | [Zhang et al., 2025](https://doi.org/10.1177/17543371251392456) uses team centroid, length, width, stretch, and related collective tactical measures. [Welch et al., 2021](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251970) studies collective states using team movement and spread. |
| Reason / decision context | We retain dline_m, cen_m, and fwd_m because they give the directly measured back, central-block, and front positions. We retain width_m and depth_m because they describe lateral and longitudinal spread. We retain speed_ms as group activity. We exclude line_gap_m here because it is calculated from fwd_m minus dline_m; including all three would repeat the same longitudinal information. |
| What it may help infer | High/low team block, wide/narrow shape, spread/compact shape, and general group activity. It cannot prove possession, player roles, or a named formation. |

## Schema 2: 02_shape_dynamics.csv

| Item | Decision |
|---|---|
| Input type | Standalone temporal team-shape input. It contains all Schema 1 fields plus short-window movement fields. |
| Fields added to Schema 1 | Home/Away line_gap_change_3s, cen_change_3s, width_change_3s, speed_change_3s, dline_fwd_max_cross_corr_5s, and dline_fwd_lag_at_max_corr_s. |
| Why this input type | A single snapshot cannot tell whether a team is stepping up, retreating, widening, or compacting. This schema adds controlled time-based information without using ball or tag context. |
| Research cited | [Welch et al., 2021](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251970) supports analysing changing collective states. [Gonçalves et al., 2016](https://pubmed.ncbi.nlm.nih.gov/27079483/) uses cross-correlation and time-lag methods for football coordination. [Corsie and Swinton, 2023](https://pubmed.ncbi.nlm.nih.gov/35838043/) shows why short-window positional metrics need careful reliability handling. |
| Reason / decision context | Three seconds is the first selected movement window because it is long enough to reduce one-frame noise but short enough to retain local tactical movement. Line-gap change is included because it has a direct football meaning: stretching or compacting. Cross-correlation and lag are included as research methods for coordinated line movement, not as a custom tactical rule. |
| What it may help infer | Team expansion, compacting, coordinated movement, stepping up, retreating, and possible transition periods. It cannot confirm the exact frame where possession changed. |

## Schema 3: 03_ball_shape_extended.csv

| Item | Decision |
|---|---|
| Input type | Standalone ball-aware team-shape input. It contains all Schema 1 fields plus ball, missing-ball, and local spatial-ball fields. |
| Direct ball fields | ball_visible, ball_x_m, ball_y_m, ball_home_adv_m, home_ball_dist_m, away_ball_dist_m, ball_margin_m. |
| Team-to-ball fields | home_cen_to_ball_offset_m and away_cen_to_ball_offset_m. These say whether each central block is behind, level with, or ahead of the observed ball. |
| Missing-ball fields | seconds_since_ball_seen, seconds_until_ball_seen, left_anchor_margin_m, right_anchor_margin_m, anchor_proxy_agrees, gap_length_s. |
| Extended spatial fields | Home/Away track counts ahead of and behind ball, local support count within 5 m, and compactness behind ball. |
| Why this input type | It tests the added value of direct ball geometry while still keeping full team shape. It can be read independently by any model; it does not need Schema 1 or 2 to be joined. |
| Research cited | The [systematic review of distance variables](https://pmc.ncbi.nlm.nih.gov/articles/PMC7143020/) identifies ball, team, player and space distances as collective tactical measures. A football pressure study uses a 5 m radius as a defensible local-pressure choice. [Mathematical pressure model, 2024](https://www.mdpi.com/2227-7390/12/23/3854) Long ball gaps must be handled as uncertainty rather than fabricated ball routes. [Miura et al., 2009](https://doi.org/10.1016/j.cviu.2008.10.005) |
| Reason / decision context | Ball margin and team-to-ball distances are kept because they say how strong the geometric ball clue is. They do not equal possession. Gap fields preserve the last and next reliable ball observations without inventing hidden ball coordinates. The extended fields are deliberately collected in this one file so we can test whether local spatial detail improves a model beyond core shape. |
| What it may help infer | Whether ball-related evidence is reliable, whether a team block is positioned ahead of or behind the ball, local support/pressure context, and whether the state may have changed during a ball gap. It cannot treat nearest_team as possession truth. |

## Schema 4: 04_full_phase_context.csv

| Item | Decision |
|---|---|
| Input type | Standalone full-context input. It contains shape, dynamics, ball/gap, extended local spatial-ball, and direct Smart Tagging fields. |
| Fields added beyond Schema 3 | line-gap, centre, width and speed changes; dline-to-fwd coordination fields; tag_poss_hint; set-piece fields; Build Up, Long Ball, Attacking Transition, and Defensive Transition tag indicators. |
| Why this input type | It tests the practical question: when automatic phase context is available, does it add useful information beyond tracking data alone? |
| Research cited | [Zhang et al., 2025](https://doi.org/10.1177/17543371251392456) compares collective tactical behaviour across in-possession, out-of-possession, and transition phases. [Borghesi et al., 2023](https://iris.polito.it/handle/11583/2976470) treats possession estimation as a temporal tracking-data problem. |
| Reason / decision context | Direct Smart Tagging is included only because it is an external feed that may be available at model runtime. It is not possession truth. This file tests whether that available context improves results over the ball-and-shape file. |
| What it may help infer | Whether the present tracking pattern also has automatic evidence of build-up, long ball, restart, or transition. It does not allow the model to use the possession answer as an input. |

## Why this design is architecture-agnostic

Each schema is a flat, standalone CSV. A tree model can use one row at a time. A temporal model can place consecutive rows into a time sequence. A Transformer can turn each row into a token. The columns do not depend on one particular model architecture.

The master feature table remains the storage layer. These schemas are controlled, complete exports for experimentation:

1. Schema 1 asks whether basic shape is enough.
2. Schema 2 asks whether movement over time adds value.
3. Schema 3 asks whether ball and ball-gap geometry adds value.
4. Schema 4 asks whether full available context adds value.

## Data deliberately not used in the first schemas

The project contains more data than these first schema versions use. The omissions below are deliberate; they are not accidental.

| Data source or field group | Used in these schemas? | Why it is not an input now | Possible later use |
|---|---|---|---|
| Tactical homography CSV | No, entirely skipped as model input. | Homography maps already-known pitch positions to video pixels. It does not add a new football-position signal to the structured tracking schemas. | Visual checking: overlay a detected team shape on the tactical video to see whether a strange value is real or a tracking defect. It can also support a future image/video model. |
| Tactical MP4 video pixels | No. | The first schemas are structured numerical inputs, not a computer-vision model. Adding pixels would create a different multimodal modelling problem. | A later video-plus-tracking architecture, after the structured baseline is understood. |
| Broadcast MP4, broadcast ATD, broadcast homography | No. | Broadcast has a different timeline, cuts, replays, and lower homography coverage. Its alignment to the tactical timeline is match-specific and has reliability tiers. | A cautious extra source during tactical ball gaps, only when its alignment tier is strong. |
| Raw player-track slot IDs and per-track x/y columns | Not directly. Used only to calculate team summaries. | A track slot is not a stable named player. Slots can disappear, duplicate, and switch people, so giving slot 1 through slot 46 directly to a model could teach tracker behaviour instead of football behaviour. | Short, visually verified local-player analysis or a permutation-invariant player-set model. |
| Individual player speed and estimated direction | Not directly. | The delivered speed is only a magnitude; it has no reliable direction. Track gaps and swaps make individual direction estimates unsafe for long windows. | Very short-window collective-flow features, after dedicated quality checks. |
| Raw ball speed | No. | The repository audit reports it is often missing and unreliable; tracker jumps can create false high speed. | Only after a separately validated ball-cleaning method. |
| Raw team-centre x/y coordinates | No. | Raw x changes football meaning when teams switch ends, and it overlaps with the direction-normalised central-block field. | A later lateral-shift or cross-pitch movement experiment, with explicit direction handling. |
| Surface area and convex hull | No in the first schemas. | Missing and duplicate tracks can strongly distort occupied-area measurements. | A later spatial-coverage experiment with strict shape-confidence thresholds. |
| Named formations and individual player roles | No. | The supplied tracking does not provide stable player identities or reliable role labels. | Only if a separate, validated role/formation model is added. |
| Player-facing direction, acceleration, pass events, ball carrier identity | Not available reliably. | These are not present as reliable input fields in the supplied tracking. | Requires a different data provider or separately validated extraction. |
| Smart Tagging details not listed in Schema 4 | No. | Examples include Side, Direction of ball entry, and Max Players in the Box. We start with the highest-level phase and set-piece context so File 4 stays interpretable. | A later tag-ablation experiment after the direct-phase context baseline is tested. |
| nearest_team as a label | Never. | It is only the team whose visible track is geometrically nearest to the ball. It can favour defenders and is not possession truth. | It remains a weak geometric clue through distance and margin fields, never a ground-truth target. |

### Why homography is skipped

Homography is useful, but it solves a different problem. It transforms a location already known on the pitch into the matching video pixel. Our schemas already use tracking positions in pitch metres, so homography would mostly duplicate the coordinate system rather than add a new tactical measurement.

We keep it for quality control:

1. Draw the Home/Away tracks and the three team-line markers onto the tactical video.
2. Check whether an unusual width, line height, or missing player is a genuine situation or a tracking problem.
3. Do not manually correct the frame or create a manual possession label from that check.

The tactical homography is valid for every tactical frame; the broadcast homography has substantial missing coverage and is therefore not part of the first structured input designs. See [DATA.md](../../../DATA.md) and the [off-the-ball data guide](../README.md).

### Raw-data concepts excluded from the first schemas

This table answers a different question from the table above: even when a football concept would be useful, does the supplied raw ATD/XML/TXT data actually provide it reliably?

| Football concept or raw signal | Status in supplied data | Used in first schemas? | Why it is excluded or limited |
|---|---|---|---|
| Team formation for Home or Away | Not supplied as a reliable label. | No. | There is no official 4-3-3, 4-4-2, or similar formation field. Formation inference would be a separate uncertain model because track slots are temporary. |
| Stable player identity or shirt number | Not supplied. Track slots are not people. | No direct slot features. | Track 1 through Track 46 can disappear, duplicate, or switch to another real player. A model could learn tracker behaviour instead of football. |
| Player role | Not supplied reliably. | No. | The data cannot reliably say that a track is a striker, midfielder, centre-back, or full-back throughout the match. |
| Facing direction / body orientation | Not supplied. | No. | The tracking gives location and speed magnitude, not where the player's body is facing. |
| Acceleration | Not supplied directly. | No. | It could be calculated from changes in delivered speed, but gaps, slot switches, and noisy speed can create false acceleration. |
| Individual pass events | Not supplied. | No. | Smart Tagging includes broad phases such as Long Ball and Build Up, not a reliable exact pass event with passer, receiver, and frame. |
| Ball-carrier identity | Not supplied. | No. | The tracking does not identify which player is controlling the ball. |
| nearest_team | Calculated from the closest visible team track to the observed ball. | Not as a target or direct possession answer. | A defender can be nearest while the opponent controls the ball; the ball can be airborne; and tracking can be wrong. We use its underlying distances and ball_margin_m only as weak geometric evidence. |
| Individual player x/y positions | Supplied for temporary track slots. | Used only indirectly. | They are aggregated into collective team shape fields. Raw slots are not passed directly to the first models because identities are unstable. |
| Individual player speed | Supplied as a magnitude only. | Used only indirectly. | We use mean team speed. We do not claim an individual player's direction, sprint, or role-specific movement. |
| Ball x/y position | Supplied only when the ball is visible. | Yes, with a visibility flag. | Ball coordinates are valuable but missing often; the model must see ball_visible and gap-age fields rather than treating a missing ball as zero or as a dead-ball state. |
| Ball speed | Supplied but unreliable. | No. | It is often missing and can contain tracker jumps, so it is not safe for the first schemas. |

In short:

    Not supplied: formation labels, roles, facing direction, pass events, ball carrier.
    Supplied but unstable: track IDs, individual tracks, individual speed over long windows.
    Supplied but unreliable: ball speed.
    Supplied as a weak clue only: nearest_team.
    Supplied and used with quality flags: ball position and team-level shape.

## Research basis

- [Existing data guide](../README.md) and [frame-table schema](../../data-layer/SCHEMA.md) define the available data.
- [Zhang et al., 2025](https://doi.org/10.1177/17543371251392456) supports team position, spread and stretch measures.
- [Gonçalves et al., 2016](https://pubmed.ncbi.nlm.nih.gov/27079483/) supports coordination through cross-correlation and lag.
- [Systematic review of distance variables](https://pmc.ncbi.nlm.nih.gov/articles/PMC7143020/) supports ball-to-team and ball-to-space measures.
