# Model-input schemas

The four schemas originally designed in PR 2 are implemented by
`src/features/model_inputs.py`. They are separate flat experiment inputs, not
four stages that must be concatenated. `src/features/classification_inputs.py`
adds the structured anonymous-player and ball views needed by set/temporal
models while preserving the flat schemas for controlled ablations.

Read the [research review](RESEARCH_REVIEW.md) and append-only
[decision log](DECISION_LOG.md) before changing this contract. For a concise
team update and PR plan, use the [team handoff](TEAM_HANDOFF.md).

| schema | columns | purpose | additions to the original header |
|---|---:|---|---|
| `01_core_shape` | 20 | per-frame team shape and quality | none |
| `02_shape_dynamics` | 32 | schema 01 plus 3 s changes and 5 s line coordination | none |
| `03_ball_shape_extended` | 60 | schema 01 plus observed-ball, gap-anchor and player-around-ball geometry | 17 tracking-only dead-ball fields |
| `04_full_phase_context` | 80 | schemas 02 and 03 combined, plus generated phase context | 17 dead-ball fields; original provider `tag_*` fields replaced by 8 `gtag_*` fields |

The authoritative ordered column lists are `SCHEMA_COLUMNS`. The public call is:

```python
policy = InputPolicy(schema=SCHEMA_PHASE, temporal_mode=TemporalMode.OFFLINE_ACAUSAL)
inputs = make_classification_inputs(table, slots, pitch, policy, fps=25)
X = inputs.frame_features
```

`frame_id` is the returned DataFrame index. It is available for row alignment
and CSV export but is deliberately not a trainable feature. Use
`standalone_frame(X)` when a file needs an explicit `frame_id` column.

Schemas 03 and 04 require the anonymous `slots` arrays. This is a correction to
the earlier two-argument pseudocode: local support, ahead/behind counts and
compactness cannot be reconstructed from the aggregate frame table. The same
slot contract is available for both Sportec and BAR-ATH.

## Leakage boundary

- Model inputs are selected from immutable per-schema allow-lists.
- Schema 04 reads only `gtag_*` tags generated from damaged tracking with the
  same code on both datasets.
- Provider Smart Tagging `tag_*`, Sportec BallPossession, labels, events,
  damage metadata, grading flags and match/team identities are not accepted.
- All rolling, lagged and gap-anchor computations stop at period boundaries.
- Missing values remain missing and are paired with visibility/confidence
  fields; they are not silently filled with zero.

Run the complete hand-off gate before any model fit:

```bash
python scripts/validate_model_inputs.py
python scripts/diagnostics_round1.py --no-reference
```

The first command builds all four schemas for all seven fixed-split Sportec
matches and checks row/label alignment, ordered columns, dtypes, constants,
all-missing features and provider-tag exclusion. The second runs the broader
data-risk diagnostics. BAR-ATH parity must also be checked when its licensed
raw bundle is available locally.
