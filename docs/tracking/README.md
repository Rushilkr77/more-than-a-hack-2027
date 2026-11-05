# Tracking feature pipeline

This module is the measurement layer for inferred Smart Tagging intervals. The IDSSE generator combines direct facts (corners, kick-offs, free kicks, goal kicks, throw-ins, and goals) with a conservative first pass of ball-aware phase rules. This layer supplies the reproducible movement evidence those rules use.

## Pipeline and responsibility boundaries

```text
provider tracking
  -> provider adapter
  -> TrackingFrames (metres, teams, attack direction, fps)
  -> TrackingFeatureEngine
  -> cached frame table
  -> tactical rule + inferred confidence
  -> Smart Tagging XML interval
```

`src/tracking.py` is the single core module. It has three deliberate responsibilities:

| component | responsibility | why it exists |
|---|---|---|
| `MetricaATDAdapter` | turns the existing More Than A Hack slot arrays into the common representation | keeps the feature code independent of the raw text format |
| `SportecTrackingAdapter` | turns Sportec XML positions into the same representation | allows the seven IDSSE matches to use the same rules later |
| `TrackingFeatureEngine` | calculates compact team shape and motion series | lets tag rules depend on football measurements, rather than data-provider fields |
| `inferred_confidence` | applies the configurable confidence penalty to a derived label | prevents an inferred tag looking as certain as a directly recorded event |

`TrackingFrames` is the boundary object. It contains only player positions in metres, team ownership of each provider slot, pitch dimensions, frame rate, and each team's attack direction. It intentionally carries no tactical conclusion.

### Participant richness for training

`TrackingFrames.participants_by_slot` is aligned one-for-one with the position-array columns. It keeps identity metadata outside Smart Tagging XML, where it belongs in the training layer:

| provider | participant key | retained metadata | identity quality |
|---|---|---|---|
| Sportec | real `PersonId` | team ID, shirt number, first/last/short name, playing position, starter status | `verified_person` |
| Hackathon ATD | `slot_0`, `slot_1`, … | team assignment only | `tracking_slot` |

This difference is non-negotiable: Sportec box counts can count distinct real people, while Hackathon counts describe visible tracklets and must carry lower confidence when duplicate tracks are plausible. Smart Tagging XML still receives only the reference-compatible tactical labels; the participant catalog remains available for model features and reproducible derivation.

## Why the coordinates are normalized first

Raw pitch x-coordinates mean opposite things when the teams change ends. For every player frame, the engine uses the explicit direction to calculate `metres_from_own_goal`:

```text
if team_attacks_toward_increasing_x:
    metres_from_own_goal = raw_x_metres
else:
    metres_from_own_goal = pitch_length_metres - raw_x_metres
```

Larger values therefore always mean *further toward that team's attacking goal*. A rule written once can apply in either half and to either provider. The names in code deliberately state the meaning (`team_attacks_toward_increasing_x`, `metres_from_own_goal`) rather than hiding it behind a short name such as `attack_x`.

## Measurements emitted for each team and frame

Only seven measurements are emitted. The five positional values describe the present shape; two rates describe whether that shape is moving. All require at least seven on-pitch tracks, otherwise positional values are `NaN` and `shape_quality` is zero.

| feature | calculation | concrete utility in a future tag rule |
|---|---|---|
| `defensive_line_position_m` | mean of the 2nd–5th deepest visible tracks | detects a high/retreating defensive block and the start of a defensive transition |
| `midfield_line_position_m` | median visible outfield position after excluding the deepest track | gives a stable, central measure of whether the team advances or retreats together |
| `forward_line_position_m` | mean of three advanced tracks after excluding the single highest one | separates a high attacking occupation from a team whose middle has moved but attack has not |
| `team_width_m` | lateral 10th–90th percentile spread | distinguishes a broad attacking shape from compact central defending without a single wide outlier deciding it |
| `shape_quality` | `clip((visible_tracks - 5) / 5, 0, 1)` after the seven-track minimum | gates rules when tracking coverage is too poor to trust a team-shape claim |
| `defensive_line_velocity_mps` | centred-window rate of defensive-line position | quantifies a press stepping up or a block dropping, rather than inferring movement from one noisy frame |
| `midfield_line_velocity_mps` | centred-window rate of midfield-line position | detects collective midfield advance/retreat, useful for build-up and transition candidates |

Velocities use a centred 12-frame window (0.48 seconds at 25 fps). Edge frames or frames whose two endpoints lack a valid shape remain `NaN`; they are not silently treated as stationary.

## Why this is not a centroid-based model

A centroid is one arithmetic average of every visible player location. It is still present in the old frame table as a backwards-compatible diagnostic (`*_cen_m`), but it is not a primary input to new tag inference.

One point cannot say whether a team has a high defensive line, a low block with advanced forwards, or a stretched shape. A forward running high and a defender dropping deep can leave the centroid unchanged while the tactical state changes materially. The three ordered lines retain that distinction, while `team_width_m` records a separate lateral dimension. This is a compact time series—not an eleven-player trajectory model—and it is therefore enough for initial team-level tag rules without inventing unreliable player identities from the Hackathon tracking slots.

## Data-provider choices

### More Than A Hack / Metrica ATD

`frame_table.build_table()` already loads ATD slots. It now adapts those arrays through `MetricaATDAdapter`, then appends the fourteen team-specific feature columns to the cached frame table. The source's home attack direction is read from its metadata and flipped for the second half, exactly as the existing table does. Cache schema `v1` ensures old cached tables are rebuilt.

### Sportec / IDSSE

`SportecTrackingAdapter` reads the position XML and its metadata pitch/team fields. Sportec player coordinates are already metres. The caller must supply `attacks_toward_increasing_x_by_section`, for example a mapping keyed by `firstHalf` and `secondHalf` whose values are `(home_direction, away_direction)` booleans.

This is intentional: the raw Sportec position XML identifies the game section but has not yet been independently validated as a reliable attack-direction declaration. The adapter refuses to load a section without an explicit direction instead of guessing and producing mirrored tactical features. A small synthetic regression test verifies that both adapters produce the same canonical convention.

The Sportec adapter now retains ball coordinates and elapsed time alongside the player series. The phase generator uses ball visibility, the nearest-player team as a deliberately weak possession proxy, and attack-relative ball territory. Ball-derived inference is kept separate from the team-shape engine so that its assumptions stay visible and independently replaceable.

## How this becomes Smart Tagging XML

The current workflow is:

1. `scripts/generate_idsse_smart_tagging.py` maps direct IDSSE event facts into Smart Tagging XML. Those records have direct-event confidence (`1.00` when a decision timestamp exists, otherwise `0.90`).
2. `scripts/generate_idsse_inferred_smart_tagging.py` resolves attack direction from the IDSSE kick-off's `TeamLeft`/`TeamRight` fields, loads the Sportec player and ball frames, and selects continuous five-second intervals. Its initial rules label ball-in-own-third possession as `BUILD UP`, ball-in-middle-third possession as `PROGRESSION`, ball-in-final-third possession as `CREATING CHANCES`, and the opponent equivalents as the three defensive-third codes.
3. The rule aggregates the relevant frame values and writes one XML interval using the same Smart Tagging field structure. It does not overwrite or duplicate a direct event interval.
4. The XML `Confidence` label for that inferred interval is calculated as:

   `clip(rule_strength, 0, 1) × clip(observation_quality, 0, 1) × inferred_confidence_multiplier`

   The initial multiplier is `DEFAULT_INFERRED_CONFIDENCE_MULTIPLIER = 0.60`, exposed by `inferred_confidence(..., multiplier=...)`. It is a policy variable, not a claim that the inferred event is 60% accurate. Increasing or decreasing it changes only derived labels; directly sourced event confidence remains independent.

5. Each added rule must be validated against a held-out set of existing Smart Tagging examples and recorded in `docs/data-layer/SMART_TAGGING_EVIDENCE.md` before it becomes a training-label generator.

## First-pass limits

- `BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, and the three defensive-third codes are weak labels, not validated ground truth. They require five uninterrupted seconds with a visible ball and a stable nearest-player team.
- No `ATTACKING TRANSITION` or `DEFENSIVE TRANSITION` labels are emitted. A raw nearest-player-team switch flickers at player-contact level and is not possession turnover evidence.
- No production `LONG BALL` rule is emitted. It needs a ball-flight/pass segmentation rule rather than a simple forward-distance threshold.
- `BALL IN FINAL THIRD`, `BALL IN THE BOX`, and `PLAYERS IN THE BOX` remain absent because Smart Tagging requires their `Side`, `Direction of ball entry`, and/or `Max Players in the box` labels. Those fields must be derived and validated together; emitting only their parent code would violate the reference schema.
- No formation classifier, player-role classifier, player identity matching, or player-specific motion model.
- No automatic direction *guessing*: direction is read from the explicit kick-off side fields.
- No validated possession, pressure, or offside calculation.
- No claim that the seven measurements improve model performance; every eventual rule needs an ablation and evidence record.

These exclusions keep the module useful rather than bloated. They can be added only when a concrete Smart Tagging field needs them and there is a testable validation path.

## Verification

Run the focused tracking tests and the full suite:

```bash
.venv/bin/python -m pytest -q tests/test_tracking.py
.venv/bin/python -m pytest -q
```

The tests cover attack-relative symmetry, centred velocity, quality and confidence math, adapter normalization, the existing frame-table schema, and the generated schema documentation.

## Reference profiling before inference

Run the profiling command after rebuilding the Hackathon frame table:

```bash
.venv/bin/python scripts/profile_smart_tagging.py
```

It writes the ignored, regenerable [`BAR-ATH_smart_tagging_profile.json`](../../outputs/profiling/BAR-ATH_smart_tagging_profile.json). The file carries all 638 reference intervals, not selected examples. Every interval has the same seven team-shape and eight ball/box measurements, its own team-and-half baseline deltas, and data-coverage values. Each code also carries a `sportec_rule_template`:

- team-shape values are interquartile ranges relative to the same team's half baseline, which controls for team style;
- ball/box values are interquartile ranges in pitch metres or fractions, which transfer directly to Sportec's metric coordinates;
- template ranges are candidate evidence only, never a requirement to force a tag during missing-ball or poor-shape observations.

This profile is the evidence source for the next inference-rule phase. It keeps the reference XML's role explicit: it calibrates an exploratory rule, rather than proving the rule generalizes to another match.
