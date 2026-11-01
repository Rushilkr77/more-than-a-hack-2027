# Off-the-ball movement research guide

## Purpose and scope

This guide describes the information available for studying **collective off-the-ball movement** in the Barcelona–Athletic match: how each team occupies space, moves as a group, and changes shape over time.

It is deliberately **not** a possession model specification. The data can reveal associations between movement and match context, but it cannot by itself prove that a movement pattern caused possession or identify a player's tactical intent.

The useful unit of analysis is normally a short time window (for example, 1–5 seconds), not one video frame. The tactical video runs at 25 frames per second, so a five-second window contains 125 frames.

```text
tracking positions + team membership + time
                    |
                    v
       team shape and team movement over a window
                    |
                    v
  context from ball visibility, tags, video, and broadcast tracking
                    |
                    v
       a football situation to inspect and describe
```

## What each source contributes

| Source | What it contains | Practical use for movement research |
|---|---|---|
| Tactical ATD XML + TXT | Player/track and ball positions and speeds at every tactical-video frame | Primary source for positions, movement, and team shape. The tactical timeline is the submission/reference timeline. |
| Tactical ATD XML metadata | Team membership for each track slot and the first-half attacking direction | Separates tracks into home/away and makes “forward” meaningful after the halftime side swap. |
| Smart Tagging XML | Automatic phase and set-piece intervals, often with a team tag | Gives contextual questions such as “what does each team’s shape look like during Build Up?” It is not ground truth. |
| Tactical homography CSV | Pitch-to-pixel transform per tactical frame | Lets a researcher overlay positions on the tactical video to verify a pattern visually. It does not create additional player-position data. |
| Broadcast ATD + homography | A second recording and tracking stream, on its own timeline | Can corroborate ball/track availability after alignment. It is optional and has cuts, replays, and lower homography coverage. |

The data bundle and its restrictions are described in [DATA.md](../../DATA.md). Do not commit or redistribute raw or derived frame-level match data.

## Important vocabulary

- **Track slot** — a temporary tracker identity, not a named footballer. Slots may disappear, be duplicated, or switch people.
- **Team shape** — a numerical summary of where a team’s visible tracks are, not a recognised formation such as 4-3-3.
- **Team-relative distance** — distance from that team’s own goal line. Larger means further forward, regardless of which direction the team attacks in that half.
- **Window** — consecutive frames analysed together to describe movement. At 25 fps, 25 frames is one second.
- **Confidence** — a value that says whether enough usable tracks or ball observations exist to trust a measurement.

## What the player tracking can and cannot say

### Available at the raw level

For each tactical frame, the ATD provides:

- normalised x/y position and speed for each visible track slot;
- normalised x/y position for the ball when it is visible;
- team membership for each slot from the metadata XML;
- frame number, where raw `frame_id - 1` equals the tactical-video frame.

Coordinates are converted to metres using the 105.3 × 68.0 m pitch. This makes distances and movement rates interpretable.

### Not available reliably

- Stable player identities, shirt numbers, or individual roles for most tracks.
- Reliable formation labels.
- A manually corrected view of the tracking.
- A ball location on every frame.
- The identity of the ball carrier or a player-level `on ball` / `off ball` flag. `nearest_team` is only a geometric proxy and is not possession.
- Player facing direction, body orientation, acceleration, pass events, or a velocity vector. The delivered player `s` field is a speed magnitude only. A direction can be estimated from adjacent x/y positions, but gaps and track switches make that unsafe for long-lived individual-run claims.

There are 46 tactical track slots for 22 on-pitch players. Therefore, prefer collective statements such as “the home defensive line advanced” over long-lived individual statements such as “the same full-back made this run.” Individual movement is safest only over short, visually checked intervals.

In this guide, **off the ball** therefore means movement of a team or a short-lived visible track while it is not being treated as the ball carrier. It does **not** mean that the data can establish an individual player's tactical intention.

## Existing team-movement measurements

`src/features/frame_table.py` builds the shared table documented in [the frame-table schema](../data-layer/SCHEMA.md). It already includes the following team-level measurements.

| Measurement | Fields | What it describes in practical terms |
|---|---|---|
| Number of visible tracks | `n_home`, `n_away` | Whether enough of each team is visible to support a collective reading. Counts can exceed 11 because the tracker may duplicate people. |
| Raw centre | `home_cx_m`, `home_cy_m`, `away_cx_m`, `away_cy_m` | The average visible position of a team in raw pitch coordinates. Useful for basic spatial comparisons, but interpret alongside attack direction. |
| Spread | `*_depth_m`, `*_width_m` | How spread the visible team is along the pitch and across its width. A change can indicate compression, expansion, or tracking loss. |
| Team pace | `*_speed_ms` | Mean delivered speed of visible tracks. Use as a broad activity measure, not as a role-specific running statistic. |
| Defensive line | `*_dline_m` | Average team-relative distance of the 2nd–5th deepest on-pitch tracks. The deepest track is excluded as a likely goalkeeper/noise track. Larger means the line is higher upfield. |
| Team centre | `*_cen_m` | Average team-relative distance of visible outfield tracks after excluding the deepest track. A summary of how high the collective block is. |
| Forward line | `*_fwd_m` | Average position of the leading tracks after trimming the single furthest-forward track. Avoids one spurious or isolated runner defining the line. |
| Vertical stretching | `*_line_gap_m` | Forward line minus defensive line. Larger means the team is more stretched between its lines. |
| Shape reliability | `*_shape_conf` | A 0–1 confidence measure based on available on-pitch tracks. Shape fields are unavailable below seven usable tracks. |

The line and centre measures use team-relative distance-from-own-goal, so they retain the same football meaning in both halves. Do not compare raw x positions across halves without applying direction.

## Movement signals to study over time

The table holds values at a frame. Movement research comes from comparing those values across a window.

| Change over a short window | Practical interpretation to investigate |
|---|---|
| Defensive line rises or drops | The block may be stepping up, pressing, recovering, or retreating. Check phase context before naming it. |
| Centre rises/drops with the defensive line | The whole team may be moving collectively, rather than only one line changing. |
| Line gap widens | The team may be stretching between defence and attack; it can also expose a transition or a tracking defect. |
| Line gap narrows | The team may be becoming compact. |
| Width opens/closes | The team may be spreading play, defending wide spaces, or collapsing toward a central threat. |
| Mean speed rises for one/both teams | A possible transition, counter-press, sprint recovery, or restart; inspect the surrounding context. |
| Track count falls | A reliability warning, not a tactical movement conclusion. |

Use differences such as “end of window minus start of window,” rolling averages, and minimum/maximum values. These describe movement without assuming that an individual track remains the same person for the full match.

### Frame and window eligibility

Build a movement window entirely within one half. Exclude halftime and `pre_kickoff` frames: people may be visible there, but they are not a normal in-play tactical state. For a ball-relative conclusion, require both a usable team shape and an observed ball at the relevant comparison points. The tactical ball is visible on only 59.3% of graded frames, and its gaps are often long, so carrying its last location forward is context only, never evidence of the ball's current position.

## Relating the two teams

Many useful movement observations only appear when home and away measures are read together.

| Combined observation | What it can describe |
|---|---|
| Home line advances while away line retreats | A territorial shift or one team driving the other back. |
| One team becomes compact while the other widens | A possible defensive block versus an attacking team using width. |
| Both teams’ centres move in the same direction | The match may be moving as one block, for example in a transition. |
| Both teams’ lines remain high/low | A sustained field-position state rather than a sudden movement event. |
| One team’s width/depth changes sharply while the other is stable | A local tactical response, or a data-quality issue worth checking on video. |
| Large difference in shape confidence | Do not make a strong two-team comparison; one side may simply be poorly tracked. |

Useful additional collective measures can be derived from raw positions when needed: distance between team centres, overlap between the teams’ longitudinal ranges, team occupied area, and density of tracks in pitch zones. Describe them as collective measurements; do not infer named player roles from them.

## Available raw-derived measures not yet in the shared table

The existing table deliberately keeps a small, robust set of shape fields. The raw tactical slots provide further **collective** measures when the research question needs them. These are available data, not already-validated conclusions; calculate them on short, quality-screened windows and retain the reliability fields alongside them.

| Measure | How to derive it | What it can help describe | Main limitation |
|---|---|---|---|
| Block-to-ball offset | Compare a team's direction-normalised centre, defensive line, or forward line with the ball's direction-normalised position | Whether a team block is ahead of, level with, or behind the ball | Requires a visible ball and usable team shape |
| Players ahead of / behind the ball | Count visible team tracks whose direction-normalised position is ahead of or behind the ball | Support behind the ball, runners beyond it, or retreating defenders | Counts tracklets, so duplicates can inflate it |
| Local support and pressure | Count home and away tracks within a declared radius, such as 3 m, 5 m, or 10 m, of the ball; retain the nearest and second-nearest distances | How crowded the ball area is and whether either team has nearby support | Geometric proximity is not controlled possession |
| Lateral relation to the ball | Compare team centre/width and track-zone counts with the ball's y position | A possible wide overload, central collapse, or far-side spacing pattern | Requires an explicit, pre-declared zone or threshold |
| Occupied area / spatial coverage | Compute a convex-hull area or a fixed pitch-grid occupancy from visible tracks | Whether a team expands, contracts, or occupies more of the pitch | Sensitive to missing/duplicate tracks; compare only similar-confidence windows |
| Direction estimated from positions | Difference consecutive visible x/y positions over a very short run | Approximate collective flow, such as a block shifting laterally | Do not use the delivered ball-speed field; it is mostly missing and unreliable, and track gaps can create false directions |

The XML marks two slots as goalkeepers, which can be a useful sanity check when reviewing a shape. Do not turn that into a stable named-player role: the slots remain automatic tracklets. The current defensive-line field instead removes the deepest visible on-pitch track each frame to limit goalkeeper/noise influence.

## Adding ball context when it is available

Off-ball movement is most useful alongside ball context, but ball context must be marked as missing or uncertain when appropriate.

| Ball-related field | How it complements movement research |
|---|---|
| `ball_visible` | Distinguishes direct ball observations from movement-only periods. Missing ball does not mean dead ball. |
| `ball_x_m`, `ball_y_m` | Locates the ball relative to both teams’ shapes and pitch territory. |
| `ball_home_adv_m`, `ball_third_home` | Lets you ask whether a team is high/compact relative to the ball and its own goal. |
| `home_ball_dist_m`, `away_ball_dist_m` | Shows each team’s nearest visible-track distance to the ball. |
| `ball_margin_m` | Shows whether one team is clearly nearer to the ball or whether the ball is contested. Small margins are weak evidence. |
| Last observed ball position and elapsed missing time | Gives context during a ball-tracking gap. Treat it as stale context, not a current ball location. |

For any ball-relative measure, reject isolated implausible ball observations before interpreting the movement around them. The delivered ball track can jump between frames and can fall outside the pitch; neither condition alone proves a stoppage.

Example research snapshot:

```text
Ball was last seen in home’s defensive third 1.5 seconds ago.
Home: compact, low defensive line, stable width.
Away: line and centre advancing, moderate width, high shape confidence.
Tag: ATTACKING TRANSITION for away.
```

This is a well-described match situation worth comparing with similar windows. It is not proof of possession and should not be converted into a hand label.

## Smart Tagging XML: useful contextual events

The Smart Tagging XML is automatic tactical annotation aligned to the tactical-video clock. Tag timestamps convert to frames with `round(seconds * 25)`.

Useful phase codes include:

- `BUILD UP`
- `PROGRESSION`
- `ATTACKING TRANSITION` and `DEFENSIVE TRANSITION`
- `DEFENDING IN DEFENSIVE THIRD`, `DEFENDING IN MIDDLE THIRD`, and `DEFENDING IN ATTACKING THIRD`
- `LONG BALL`
- `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `PLAYERS IN THE BOX`, and `CREATING CHANCES`
- `SET PIECES`: throw-ins, corners, goal kicks, free kicks, and kickoffs

The tag table also exposes active tag counts, team-tag counts, and set-piece type/team in the shared frame table. These are best used to group and compare movement windows:

| Question | Data to relate |
|---|---|
| How does a team’s block look during build-up? | Team lines, width, gap, speed + `BUILD UP` tag and its team tag. |
| How does defensive compactness change in a transition? | Before/after movement of both teams + transition tag. |
| What happens around a long ball? | Both teams’ line gaps and speeds + `LONG BALL` tag. |
| How do teams arrange themselves at restarts? | Shape/track density + set-piece type and team tag. |

Tags are not possession ground truth: they are automatic, boundaries are approximate, and 36 intervals—including every throw-in—have `Team = N/A`.

## Video and homography: visual verification

The tactical homography is valid on every tactical frame. It maps tracking pitch coordinates to video pixels, so a researcher can overlay track positions and team-shape summaries on the corresponding video frame.

Use it to answer questions such as:

- Is the apparent high line real, or did the tracker duplicate a player?
- Does a sharp width change correspond to a genuine wide attack?
- Are missing tracks caused by a camera/framing problem?

Visual checking should validate an automatically measured pattern. It must not become manual correction or manual possession labelling of the supplied match.

## Tactical and broadcast/telecast time alignment

The tactical video is the reference timeline: its 199,319 frames are the rows of every submission and frame-table record. The broadcast/telecast feed has its own 164,488-frame timeline and has no organiser-provided direct frame mapping.

The data layer infers an alignment by matching the two feeds’ player configurations in pitch coordinates. For this one match, the measured mapping is:

```text
first half:  tactical frame = broadcast frame + 4,014
second half: tactical frame = broadcast frame + 26,920
```

This is an inferred, match-specific relationship, not a universal conversion. Broadcast cuts and replays create mismatches, so the cross-check assigns a reliability tier:

| Tier | Meaning | Use for movement research |
|---|---|---|
| 1 | Alignment and ball agreement are strong | Reasonable corroborating source. |
| 2 | Mostly aligned or ball agreement unavailable | Cautious supporting context only. |
| 3 | Large mismatch | Do not treat as reliable corroboration. |
| 4 | Cannot verify alignment | No usable broadcast conclusion. |

The broadcast is most useful when tactical tracking has lost the ball and the alignment tier is 1 or 2. It does not replace the tactical timeline, and the alignment constants must be re-fitted for another match. See [BROADCAST.md](../data-layer/BROADCAST.md) for the validation evidence and caveats.

## Recommended research record for one window

For each interesting 1–5 second window, record the following rather than a single unqualified claim:

```text
Time and duration:
  tactical frame range; period; whether the interval is graded

Reliability:
  home/away shape confidence; track counts; ball visible/missing;
  broadcast alignment tier, if broadcast is used

Home movement:
  defensive line, centre, forward line, line gap, width, depth,
  average speed, and their change across the window

Away movement:
  the same measurements and changes

Relationship:
  which team advanced/receded; whether each compressed or stretched;
  whether the teams moved together or in opposite directions

Context:
  ball territory and nearest-team margin if visible; Smart Tagging phases;
  set-piece/restart state; video overlay check if needed
```

## Reliability and interpretation rules

1. Use `*_shape_conf` and track counts before trusting any shape conclusion.
2. Use team-relative fields for attack/defence interpretation; raw x values change meaning after halftime.
3. A missing ball is a data-quality state, not a dead-ball label.
4. A small `ball_margin_m` means the nearest-team signal is contested; do not call it strong possession evidence.
5. Treat Smart Tagging as context, not truth.
6. Treat broadcast alignment as an explicitly graded, optional source.
7. Do not infer stable player roles or formation labels from track slots.
8. Avoid claims of causality from one match. Record movement patterns as hypotheses and validate them on permitted labelled matches.

## Repository entry points

- Raw tactical/broadcast paths and metadata: `games.py`, `DATA.md`
- Raw ATD loader: `src/data/atd.py`
- Shared per-frame table and shape definitions: `src/features/frame_table.py`
- Table column contract: `docs/data-layer/SCHEMA.md`
- Measured data quality findings: `docs/data-layer/FINDINGS.md`
- Broadcast alignment and reliability: `docs/data-layer/BROADCAST.md`
- Optional visual inspection helpers: `starter/load_homography.py`, `starter/utils.py`

