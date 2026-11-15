# Common Input Contract Design

## Goal

Make the shared Sportec layer provide the complete, leak-safe tracking contract needed by every P3-and-later approach: flat Schema 3/4 models and structured temporal models.

## Scope

The common layer will load the complete approved tracking representation for one match, preserve it through training augmentation, and call the existing PR #7 input builders. It will not add a model, alter labels, change the split, or change the scorer.

## Canonical match contract

`src.common.sportec` will expose one immutable `MatchData` value with:

- `table`: the full frame table returned by `ft.bundle_table`;
- `slots`: all anonymous tracking arrays returned by `ft.bundle_table` (`x`, `y`, `s`, `team`);
- `labels`: the existing separate `raw` and `adjusted` label arrays;
- `pitch`: the `(length, width)` tuple from `ft.bundle_args`;
- `fps`: the frame rate from `ft.bundle_args`.

The input builder receives only `table`, `slots`, `pitch`, and `fps`. It never receives `labels`. Labels remain available to the training and scoring caller after the model inputs have been constructed.

## Input selection

Each experiment declares an existing `classification_inputs.InputPolicy` before building inputs. The policy selects one of the four immutable flat schema allow-lists and its causal/offline temporal mode. The common layer provides:

```python
def classification_inputs(match: MatchData, policy: InputPolicy) -> ClassificationInputs:
    return ci.make_classification_inputs(
        match.table, match.slots, match.pitch, policy, fps=match.fps,
    )
```

This delegates to `make_classification_inputs`. Tree/flat approaches consume `frame_features`; structured CNN, TCN, GRU, and Transformer approaches consume the returned player and ball arrays; hybrid approaches consume both. No model reads the raw table or slots directly.

## Training augmentation

`mirrored` will take and return `MatchData`, rather than a separate table, labels, and pitch length. It will preserve the established `ft.swap_sides` meaning: Home and Away are exchanged while the raw pitch coordinate system remains unchanged. The slot arrays must therefore retain `x`, `y`, and `s` values and swap only their `team` codes. H/A labels are swapped.

This keeps raw player-to-ball distances unchanged and matches the current table transform, which leaves `ball_x_m` and `ball_y_m` unchanged while swapping team-relative fields and attack direction. A physical x-axis reflection is out of scope because it would require a different transform for the table's raw ball coordinates and attack direction.

## Validation

Tests will prove that:

1. `load` returns frame-aligned slots plus pitch dimensions and FPS.
2. The shared input helper builds Schema 4 and structured inputs from one loaded match.
3. The input helper does not expose labels to `make_classification_inputs`.
4. Mirroring swaps H/A labels and slot team codes but preserves slot coordinates.
5. Mirroring twice restores the original `MatchData` exactly.
6. Schema 3/4 slot-derived features remain consistent after augmentation.

`docs/common/COMMON_LAYER.md` will document that P3-and-later approaches use this shared entry point and may not reload bundles or construct slots independently.
