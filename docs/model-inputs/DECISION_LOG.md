# Input-layer decision and event log

This is append-only. A decision is not silently rewritten when evidence
changes; add a superseding entry and link the earlier ID. “Evidence” means
repository measurement or a cited source. “Assumption” is labeled separately.

## Chain of events

| Date | Event | Evidence / outcome |
|---|---|---|
| 2026-11-08 | PR 4’s first implementation was challenged because Sportec BallPossession produced Smart Tagging inputs while also producing H/A targets. | Direct target-to-input path identified; Sportec scores could be inflated and fail on BAR-ATH. |
| 2026-11-08 | PR 4 was reworked and merged. | `src/features/tagger.py` now reads tracking table/slots only and generates `gtag_*` identically for both datasets. |
| 2026-11-10 | All seven Sportec XML matches were converted into damaged BAR-ATH-format bundles. | Fixed 4/2/1 split; approximately one million frames; labels stored outside bundles. |
| 2026-11-10 | PR 2’s four flat schema designs were implemented and corrected for the merged data contract. | 20/32/60/80 features; schema 04 uses `gtag_*`; explicit allow-lists; all seven matches pass preflight. |
| 2026-11-10 | Research audit compared the flat layer with direct possession-estimation literature. | Flat schemas retained as baselines; object trajectories and permutation invariance identified as a material missing capability. |
| 2026-11-11 | Consolidated flat + structured contract implemented and validated. | `classification_inputs.py`; object permutation, missingness and causal-policy tests; all seven Sportec matches passed. |

## Decisions

### IL-001 — Target contract: team state H/A/D

- **Status:** accepted.
- **Decision:** predict Home control, Away control, or Dead/stopped per frame.
- **Evidence:** the competition target and Sportec label files use H/A/D. The
  Borghesi paper studies the same three-button game-state contract.
- **Rejected alternative:** infer a player owner first. Stable player identity
  is not available in BAR-ATH-style slots and would solve a harder, different
  problem.
- **Invalidated if:** evaluation changes to player-level possession.

### IL-002 — Hard source boundary

- **Status:** accepted, non-negotiable.
- **Decision:** model-input builders may read only frame tables, anonymous slot
  arrays, pitch dimensions, frame rate and an explicit experiment policy.
- **Excluded:** labels, BallPossession, events XML, provider `tag_*`, damage
  metadata, match/team names and identities.
- **Evidence:** the original PR 4 path demonstrated near-direct answer copying.
  Scikit-learn’s leakage guidance defines unavailable-at-prediction information
  as leakage.
- **Implementation:** immutable schema allow-lists and function signatures;
  label alignment lives only in the validation/training boundary.

### IL-003 — Match-level separation

- **Status:** accepted, non-negotiable.
- **Decision:** never split or cross-validate random frames. Preserve the fixed
  4 train / 2 validation / 1 test matches; use match groups for any internal
  sensitivity analysis.
- **Evidence:** adjacent frames are strongly dependent; both relevant football
  studies split by game. `GroupKFold` provides non-overlapping groups.
- **Invalidated if:** independent experimental units replace continuous match
  frames.

### IL-004 — Retain flat schemas, but demote them from “final representation”

- **Status:** accepted; supersedes the assumption that schema 04 alone is the
  best final input.
- **Decision:** schemas 01–04 remain controlled, interpretable ablations and the
  tree-model input. They are one view inside the consolidated layer.
- **Evidence:** collective geometry is established in football research, but
  the closest possession study benefits from object trajectories and local
  interactions that aggregates discard.
- **Trade-off:** two views add integration work but avoid forcing all model
  families into one lossy CSV.

### IL-005 — Add anonymous structured player trajectories

- **Status:** accepted.
- **Decision:** expose `[frame, anonymous slot, channel]` player values plus
  team, observation, pitch-validity, elapsed-time and goalkeeper-proxy arrays.
- **Evidence:** Borghesi et al. use per-object TCN embeddings; nearest-player
  trajectories are highly informative for possession.
- **Rejected alternative:** raw slot IDs or a fixed player order. Broken and
  duplicate tracklets make identity unstable and encourage tracker shortcuts.
- **Invalidated if:** stable, verified player identities and roles become
  available on both datasets.

### IL-006 — Require permutation-invariant player aggregation

- **Status:** accepted, model-side contract.
- **Decision:** models consuming anonymous slots must use symmetric pooling,
  set attention, or a documented dynamic ordering. Slot permutation is a
  required invariance test.
- **Evidence:** Deep Sets; Set Transformer; the direct possession paper.
- **Trade-off:** sum pooling is simpler and lower variance; attention can model
  interactions but is more data-hungry. Both must be compared.

### IL-007 — Use fixed physical coordinate normalization

- **Status:** accepted.
- **Decision:** normalize x to `[-1,1]` in a single home-attack coordinate frame
  and y to `[-1,1]` around pitch centre. Do not fit coordinate scaling on any
  match.
- **Evidence:** both direct football studies normalize direction across halves;
  pitch dimensions provide a dataset-independent physical transform.
- **Trade-off:** the home coordinate frame is not itself team-swap invariant;
  swap augmentation and tests remain required.

### IL-008 — Include short backward velocity, never bridge gaps

- **Status:** accepted as an ablation-capable channel.
- **Decision:** derive vx/vy only from consecutive observations inside one
  period. Leave it missing at gaps and boundaries; retain delivered player
  speed separately.
- **Evidence:** velocity materially helps dead/alive classification in the
  direct possession study. Repository evidence shows long gaps and tracker
  jumps, so cross-gap differences are invalid.
- **Rejected alternative:** long-window player interpolation or acceleration.
  Broken tracklets make derivatives across gaps unsafe.

### IL-009 — Missingness is explicit; long ball interpolation is rejected

- **Status:** accepted.
- **Decision:** preserve NaN and provide observation masks and time since last
  observation. Offline mode may expose time until the next observation, but no
  hidden coordinates are fabricated.
- **Evidence:** GRU-D supports mask + elapsed-time inputs. Sportec has 35–50%
  ball missingness and multi-second gaps. The Borghesi interpolation choice was
  made on games selected for low ball missingness and is not transferable.
- **Invalidated if:** a separately validated imputer demonstrates held-out
  gains and passes synthetic-mask robustness without using labels/test data.

### IL-010 — Use only a goalkeeper proxy; do not invent roles

- **Status:** accepted with low confidence.
- **Decision:** expose the deepest on-pitch track per team as a goalkeeper
  proxy, including ties caused by duplicates. Treat it as an ablation, not
  ground truth.
- **Evidence:** role information improves possession classification in the
  direct study, but the supplied data has no stable roles or identities.
- **Trade-off:** the proxy can be wrong during tracking loss or goalkeeper
  movement. Removing it is a mandatory ablation if it appears dominant.

### IL-011 — Temporal causality must be named

- **Status:** accepted, non-negotiable.
- **Decision:** competition inference is `offline_acausal`; live inference is a
  separate `causal` policy. Current causal mode permits only core flat features
  plus backward-only structured trajectories. It refuses schemas containing
  future anchors, forward windows, symmetric correlations or smoothed tags.
- **Evidence:** the direct possession paper explicitly distinguishes causal and
  acausal prediction and reports why future motion helps detect stoppages.
- **Trade-off:** offline accuracy may be higher, but predictions cannot be
  marketed as live.

### IL-012 — `gtag_*` is an optional proxy block, never truth

- **Status:** accepted with mandatory ablation.
- **Decision:** generated tags remain only in schema 04. Every final comparison
  must include the same model without the complete tag block.
- **Evidence:** code provenance proves the tags are tracking-only; observed
  agreement with raw ownership is 67.3–79.7%, not answer copying. However,
  `gtag_poss` is still a hand-smoothed nearest-team proxy and can create an
  avoidable shortcut.
- **Invalidated if:** any target/provider source is reintroduced; then remove
  the block entirely.

### IL-013 — Keep ball and player views separate

- **Status:** accepted.
- **Decision:** expose ball trajectory and quality separately from the player
  set, then let the model combine them.
- **Evidence:** direct ablations find ball motion especially important for
  dead/alive, while nearest players and team/role are important for ownership.
- **Trade-off:** this supports task-specific heads without hard-coding which
  features each head must ignore.

### IL-014 — Compare ternary and conditional two-head outputs later

- **Status:** experiment required; not decided by the input layer.
- **Decision:** the consolidated inputs must support both one H/A/D head and a
  shared encoder with Dead/Alive plus Home/Away-when-alive heads.
- **Evidence:** the direct paper’s strongest model is two-branch attention, but
  single-branch models can be competitive and our data distribution differs.
- **Reason not finalized:** this is a model/loss decision, not an input fact.

### IL-015 — All learned preprocessing is train-only

- **Status:** accepted, non-negotiable.
- **Decision:** the input builder performs only fixed physical transforms.
  Scaling, imputation, feature selection, calibration and learned masking are
  fit on training matches and frozen before validation/test/BAR-ATH.
- **Evidence:** standard leakage prevention; repository contract.

### IL-016 — No single “perfect” input is declared before ablation

- **Status:** accepted.
- **Decision:** choose the final view only after per-match ablations, robustness
  masking, symmetry, slot-permutation and causal/acausal comparisons.
- **Evidence:** the closest paper’s preprocessing assumes much lower missingness;
  our training set contains only four matches. Transfer is uncertain.
- **Rejected alternative:** adding every plausible feature. More channels raise
  shortcut, variance and maintenance risk without demonstrated benefit.

### IL-017 — Preserve raw flat coordinates only for the original ablation

- **Status:** accepted with mandatory symmetry check.
- **Decision:** keep `ball_x_m`, `ball_y_m` and `home_attacks_right` in the PR 2
  flat schemas for backward-compatible experiments. The structured view uses
  direction-normalized coordinates. Do not assume raw coordinates belong in the
  final model.
- **Evidence:** both direct football studies normalize direction; raw field
  orientation can encode half/provider conventions. The flat schema also
  carries `ball_home_adv_m`, which is direction-normalized.
- **Required test:** compare the winning flat model with raw coordinates and
  direction flags removed; run home/away swap and lateral-reflection tests.

## Open decisions

| ID | Question | Evidence needed |
|---|---|---|
| OPEN-01 | Sum pooling or set attention? | Grouped validation, parameter-matched comparison. |
| OPEN-02 | Ternary or two-head loss? | Per-match spellF1/macro-F1 plus calibration. |
| OPEN-03 | All players or nearest five? | Dynamic nearest-k ablation, including long-ball sequences. |
| OPEN-04 | Exact window and stride? | Start at 3 s past + 1 s future; sweep on validation only. |
| OPEN-05 | Keep goalkeeper proxy? | Remove-channel ablation and tracker-damage slices. |
| OPEN-06 | Keep any `gtag_*`? | Whole-block and per-tag ablations; reject if cross-dataset robustness falls. |
| OPEN-07 | BAR-ATH parity | Requires licensed raw BAR-ATH bundle locally. |
