# Smart Tagging evidence chain

This note records how we established the Smart Tagging XML contract before attempting to map event data from other football matches. It contains no licensed match data or private Drive links.

> **Superseded in part (2026-11-08):** the generators described in the 2026-11-05/06 sections (events-based, `BallPossession`-based, `src/tracking.py`) were removed because they leak the label. The XML contract and audit below still hold. See the last section, "Rework".

## Scope and source

- **Question:** Can external football event data be converted into a Smart Tagging-compatible XML dataset for model training?
- **Reference source:** the user-provided, licensed `BAR-ATH_tactical_smart_tagging.xml`, stored locally only in the gitignored `data/tactical/` directory.
- **Accessed:** 2026-11-05.
- **Confidence:** verified by parsing the actual XML, not inferred solely from repository documentation.

## What was checked

The standard-library parser in `scripts/smart_tagging_audit.py` reads only `file > ALL_INSTANCES > instance`. It deliberately ignores `ROWS`: that section is a colour/display table, not match annotation data.

The audit checks XML validity, required fields, finite and positive time ranges, known codes and label groups, duplicate labels, unique IDs, and label completeness. `Team = N/A` is recorded as a warning rather than malformed XML because it cannot train a team-specific target.

Run it with:

```bash
python3 scripts/smart_tagging_audit.py data/tactical/BAR-ATH_tactical_smart_tagging.xml --report /tmp/smart-tagging-audit.json --fail-on-error
```

## Verified results

- 638 `instance` records; IDs are unique and sequential.
- 0 structural errors: all instances have usable times, codes, required labels, and valid known values.
- 36 `team_not_available` warnings: all 26 throw-ins, 6 kick-offs, 2 `CREATING CHANCES`, and 2 `DEFENDING IN DEFENSIVE THIRD` phases use `Team = N/A`.
- XML order is **not chronological**. Consumers must sort on numeric `start`; they must not infer match order from `ID` or document position.
- Annotation times are seconds on the tactical-video clock. The repository contract converts them with `round(seconds * 25)` to tactical video frames; ranges are used as half-open `[start_frame, end_frame)` intervals.
- Durations are interval labels, not atomic event times: minimum 5.0 s, median 22.82 s, maximum 250.72 s.

## Observed XML contract

```xml
<instance>
  <ID>...</ID>
  <start>seconds</start>
  <end>seconds</end>
  <code>TACTICAL CODE</code>
  <label><group>Team</group><text>...</text></label>
  <label><group>Half</group><text>...</text></label>
</instance>
```

All codes carry `Team` and `Half`. Code-specific labels are:

| code | extra labels |
|---|---|
| `SET PIECES` | `Type` |
| `BALL IN FINAL THIRD` | `Side`, `Max Players in the box` |
| `BALL IN THE BOX` | `Direction of ball entry`, `Max Players in the box` |
| `PLAYERS IN THE BOX` | `Max Players in the box` |
| all other observed codes | none |

The observed codes are `BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, three `DEFENDING IN ... THIRD` codes, `ATTACKING TRANSITION`, `DEFENSIVE TRANSITION`, `SET PIECES`, `GOALS`, `LONG BALL`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, and `PLAYERS IN THE BOX`.

## Decision context for external event data

1. Map direct events first: goals and restart types are likely candidates for `GOALS` and `SET PIECES` intervals.
2. Treat point events and Smart Tagging intervals as different objects. Build-up, progression, transitions, territorial phases, and box-presence labels require deterministic interval-building rules and, for some labels, tracking/location data.
3. Preserve unknowns rather than inventing labels. If an external feed lacks actor team, coordinates, or a reliable clock conversion, emit an audit warning and exclude that target from supervised training until a rule is agreed.
4. Validate every generated XML with the auditor before it becomes a training input. Record the source schema, mapping rules, audit report, and any exclusions in a new dated section of this file.

## Next evidence to add

When the external event feed arrives, document its file format, period/clock convention, event and qualifier taxonomy, coordinate system, direct mappings, derived-interval rules, unmatched fields, and its audit outcome. That is the decision point for determining which Smart Tagging targets can be trained without unsupported assumptions.

## IDSSE direct-event generator (2026-11-05)

The IDSSE pilot event XML was inspected against its metadata XML. Direct `KickOff`, `ThrowIn`, `CornerKick`, `GoalKick`, `FreeKick`, and successful `ShotAtGoal` records are sufficient to generate synthetic `SET PIECES` and `GOALS` intervals without downloading tracking data. IDSSE event order is not chronological, so generation sorts parsed timestamps. A `Foul` is not exported; only an actual `FreeKick` yields a `Free Kick` set piece.

`scripts/generate_idsse_smart_tagging.py` emits one compatible XML per match. It writes `Confidence` in each annotation as the intentional XML extension: `1.00` when the direct restart carries a decision timestamp and `0.90` when the interval starts at its execution timestamp. The source is event-derived synthetic annotation, not Metrica Smart Tagging ground truth.

The seven-match direct-event run produced the following structurally valid files in ignored `outputs/idsse/`:

| match | annotations | confidence `1.00` / `0.90` |
|---|---:|---:|
| `J03WMX` | 103 | 96 / 7 |
| `J03WN1` | 104 | 96 / 8 |
| `J03WOH` | 100 | 91 / 9 |
| `J03WOY` | 112 | 102 / 10 |
| `J03WPY` | 94 | 90 / 4 |
| `J03WQQ` | 99 | 95 / 4 |
| `J03WR9` | 92 | 86 / 6 |

Every generated file passed `scripts/smart_tagging_audit.py` with zero errors and zero warnings. These counts cover `SET PIECES` and `GOALS` only; all tactical-phase codes remain intentionally absent.

## J03WPY source-to-output validation (2026-11-05)

The regenerated `J03WPY_smart_tagging.xml` was compared with an independent parse of its raw IDSSE event XML and metadata XML, rather than by reusing the generator's result. All **94 of 94** output annotations matched the selected source event in chronological order: 93 `SET PIECES` and 1 `GOALS`. The output's 14 `ROWS` colour entries also exactly match the Hackathon reference XML.

| output field | validated meaning |
|---|---|
| `ID` | New sequential output ID after chronological sort. It is deliberately not the IDSSE `EventId`; source IDs are not chronological. |
| `start`, `end` | Seconds since `General@KickoffTime`. `start` is the source `DecisionTimestamp` when present, otherwise `EventTime`; `end` is `EventTime + 5 s`. |
| `code`, `Type` | Direct mapping only: source restart elements become `SET PIECES` with the corresponding type; `ShotAtGoal/SuccessfulShot` becomes `GOALS`. |
| `Team` | Metadata `TeamName`, joined from the source team ID. |
| `Half` | Derived from the source second-half kick-off timestamp. |
| `Confidence` | Synthetic provenance score: `1.00` with an explicit decision time, `0.90` otherwise. |

### Player and shirt-number semantics

Sportec metadata contains 40 real player `ShirtNumber` values for J03WPY. These identify the actual players. The Hackathon ATD's `ShirtNumber`, by contrast, is a tracker slot index and explicitly does **not** identify a player. Neither player identity nor shirt number exists in the reference Smart Tagging XML schema, so the generated XML correctly omits both fields. Adding them would no longer mimic the target annotation format and would mix two incompatible meanings of `ShirtNumber`.

## Tracking measurement layer (2026-11-05)

The first inference prerequisite is implemented in `src/tracking.py`; its complete design record is [`docs/tracking/README.md`](../tracking/README.md). It emits a compact, provider-neutral time series for each team: defensive, midfield, and forward line positions; team width; shape quality; and defensive/midfield line velocities. These are measurements only, not new Smart Tagging labels.

The decision is to use ordered team lines rather than a centroid as the primary tactical representation. A centroid is one average point and can remain stationary when a defensive line drops while forwards advance. The selected lines preserve that distinction, and width preserves a separate lateral signal. The legacy centroid remains only for backwards-compatible diagnostics in the Hackathon frame table.

Both the Metrica ATD slots and Sportec XML adapt into `TrackingFrames` before the shared feature engine runs. Sportec's attack direction must be supplied explicitly per `GameSection`; the current raw XML inspection establishes sections and metric coordinates but does not prove a direction convention, so the adapter refuses to guess. Sportec coordinates are centre-origin metres and are shifted to the shared pitch-origin convention at the adapter boundary. Shape values require seven visible tracks and velocities are centred 12-frame differences; insufficient observations are `NaN`, not zero.

Any inferred XML interval uses `inferred_confidence(rule_strength, observation_quality, multiplier)`, initially with `DEFAULT_INFERRED_CONFIDENCE_MULTIPLIER = 0.60`. This confidence is intentionally separate from the 1.00/0.90 confidence of directly mapped IDSSE events. The first-pass rules and their limits are recorded below; each new rule still needs observed Smart Tagging examples, thresholds, exclusions, and a validation result appended here.

## Full reference profile (2026-11-05)

`scripts/profile_smart_tagging.py` now produces ignored `outputs/profiling/BAR-ATH_smart_tagging_profile.json` from all 638 reference intervals and the rebuilt frame table. It uses one fixed measurement contract for every code: seven team-shape signals (three lines, width, two velocities, shape quality) and eight ball/box signals (visibility, attack-relative territory, forward displacement, third/box/central shares, and player-box occupancy). Each interval records its difference from the same team and half's non-pre-kickoff baseline.

The report includes an explicit `sportec_rule_template` per code. Its team-shape component is normalized relative to a team's own half, while its ball/box component remains in metres or fractions for cross-provider use. These ranges are descriptive candidate evidence, not fitted production rules; missing ball data, unknown-team annotations, and tracker-slot duplication remain visible as nulls or coverage values instead of becoming invented facts.

## Participant provenance for training (2026-11-06)

The canonical `TrackingFrames` object now carries `participants_by_slot`, aligned to its position-array columns. Sportec retains `PersonId`, team ID, shirt number, name fields, playing position, and starter status with identity quality `verified_person`. The Hackathon adapter instead emits `slot_N` keys with identity quality `tracking_slot`; its XML `ShirtNumber` describes a tracking slot, not a person. This lets model features use the rich Sportec roster while preventing player-box counts or role features from treating Hackathon tracklets as verified people.

## Ball-aware first-pass phase generation (2026-11-06)

The missing input step is now implemented: `SportecTrackingAdapter` reads the `TeamId="BALL"` frame series and aligns it to player frames, retaining normalized ball x/y coordinates, raw `BallStatus`, provider `BallPossession`, game section, and seconds from kick-off in `TrackingFrames`. Raw Sportec positions are centre-origin, so the adapter shifts x/y by half the pitch dimensions before any shared feature runs. `attack_directions_from_events()` reads `KickOff@TeamLeft` and `KickOff@TeamRight`; this is an explicit feed declaration, not a direction guess. It makes ball territory comparable across both halves.

`scripts/generate_idsse_inferred_smart_tagging.py` combines the direct-event generator with conservative, five-second ball-visible phase intervals. Sportec's continuous `BallPossession` code is used directly (`1` home, `2` away, checked against the opening kick-off of all seven matches); nearest-player-to-ball is a fallback only when that source field is unavailable. `BallStatus=1` gates in-play intervals, and source `GameSection` determines `Half`. Ball position relative to the possessing team's attacking goal maps to `BUILD UP` (own third), `PROGRESSION` (middle third), and `CREATING CHANCES` (final third); the equivalent opponent states map to the corresponding `DEFENDING IN ... THIRD` label. The interval confidence is the product of a fixed rule strength, observed team-shape quality, and the configurable inferred-confidence multiplier.

For the downloaded full J03WPY Sportec position feed, the command below produced `outputs/idsse/J03WPY_inferred_smart_tagging.xml` with **488 structurally valid annotations**: 94 direct records and 394 inferred phase records. The auditor reported **0 errors and 0 warnings**.

```bash
.venv/bin/python scripts/generate_idsse_inferred_smart_tagging.py \
  --events data/idsse/pilot/J03WPY_events.xml \
  --metadata data/idsse/pilot/J03WPY_metadata.xml \
  --positions data/idsse/pilot/J03WPY_positions.xml \
  --output outputs/idsse/J03WPY_inferred_smart_tagging.xml

.venv/bin/python scripts/smart_tagging_audit.py \
  outputs/idsse/J03WPY_inferred_smart_tagging.xml --fail-on-error
```

The output code counts are 73 `BUILD UP`, 84 `PROGRESSION`, 40 `CREATING CHANCES`, 40 `DEFENDING IN DEFENSIVE THIRD`, 84 `DEFENDING IN MIDDLE THIRD`, 73 `DEFENDING IN ATTACKING THIRD`, 93 `SET PIECES`, and 1 `GOALS`.

The previous nearest-player proxy agrees with Sportec's supplied possession on only 74.76% of provider-labelled J03WPY frames, disagreeing on 36,906 frames; it is therefore no longer used when provider possession exists. No transition label is produced: a player-contact-level possession estimate is too noisy to represent a turnover. `LONG BALL` and the three ball/box codes are also deliberately absent. The ball/box codes are not optional parent tags: the reference schema requires `Side`, `Direction of ball entry`, and/or `Max Players in the box`. They need separate, evidence-backed sublabel derivations before becoming training labels.

## Rework: one tracking-only tagger for every match (2026-11-08)

The phase records above used Sportec's `BallPossession`, which is the training label, so a tag's team *was* the
answer (risk A1 in `SPORTEC.md`). They also read the clean ball (A4) and copied exact restarts from human events
(A5), and `TrackingFrames` carried PersonId and shirt numbers (A2). On BAR-ATH the model would have seen Metrica's
tags instead, which come from a different and unknown algorithm.

Changes:
- Removed: `src/data/idsse_phase_inference.py`, `src/tracking.py`, both `generate_idsse_*` scripts, the events-based
  `generate()`, their tests and `docs/tracking/README.md`. All were added by this PR; no base-repo file was deleted.
- New: `src/features/tagger.py` generates the Smart Tagging codes from the frame table only, identically on the Sportec
  bundles and on BAR-ATH. `build_table` exposes the result as `gtag_*`, and `scripts/generate_smart_tagging.py` writes
  it as XML that passes this file's audit with zero errors. Metrica's tags stay smoother-only.
- Kept: the XML contract, the audit, the writer and the reference profile.

Rules, sources and measured quality are in `SPORTEC.md`, section "Generated Smart Tagging".
