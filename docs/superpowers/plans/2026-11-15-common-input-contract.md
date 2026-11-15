# Common Input Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let every P3-and-later approach load the complete tracking contract and build its leak-safe flat and/or structured inputs through the shared common layer.

**Architecture:** Replace the `(table, labels, length)` return from `src.common.sportec.load` with an immutable `MatchData` record containing the frame table, anonymous slots, labels, complete pitch dimensions, and FPS. Add shared input construction and paired mirroring so model branches cannot drop slots or augment the table without matching slots.

**Tech Stack:** Python 3.8+, NumPy, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-common-input-contract-design.md`

## Global Constraints

- Keep labels out of `classification_inputs`.
- Reuse `InputPolicy` and `make_classification_inputs`; do not add a second feature-selection system.
- Preserve `ft.swap_sides`: swap teams/team-relative fields while raw pitch coordinates remain unchanged.
- Mirroring swaps slot team codes but leaves slot x/y/s unchanged.
- Do not change the fixed split, labels, scorer, smoother, or model schemas.

---

### Task 1: Add shared match and input interfaces

**Files:**

- Modify: `src/common/sportec.py`
- Test: `tests/test_common_sportec.py`

**Interfaces:**

- `MatchData(table, slots, labels, pitch, fps)`
- `load(match) -> MatchData`
- `classification_inputs(match, policy) -> ci.ClassificationInputs`

- [ ] Write failing tests that monkeypatch `ft.bundle_table` to return `(table, slots)` and `ft.bundle_args` to return `{"pitch": (105.0, 68.0), "fps": 25}`. Assert that `sportec.load("M")` retains the exact table and slots, returns the full pitch tuple and FPS, and reads both label arrays.
- [ ] Run `python -m pytest tests/test_common_sportec.py -q`; confirm the tests fail because `MatchData` and the new return contract do not exist.
- [ ] Implement:

```python
@dataclass(frozen=True)
class MatchData:
    table: pd.DataFrame
    slots: Mapping[str, np.ndarray]
    labels: Mapping[str, np.ndarray]
    pitch: Tuple[float, float]
    fps: int

def classification_inputs(match: MatchData, policy: ci.InputPolicy) -> ci.ClassificationInputs:
    return ci.make_classification_inputs(
        match.table, match.slots, match.pitch, policy, fps=match.fps,
    )
```

- [ ] Run `python -m pytest tests/test_common_sportec.py -q`; confirm the interface tests pass.

### Task 2: Mirror slots with the common match

**Files:**

- Modify: `src/common/sportec.py`
- Test: `tests/test_sportec_bundles.py`

**Interfaces:**

- `mirrored(match: MatchData) -> MatchData`

- [ ] Write a failing test using the existing synthetic bundle fixture. Construct `MatchData`, call `sportec.mirrored`, and assert that x/y/s arrays are unchanged, `team` equals `1 - original_team`, raw/adjusted H/A labels swap, and the pitch/FPS are unchanged.
- [ ] Run `python -m pytest tests/test_sportec_bundles.py::test_common_mirror_keeps_slot_geometry_and_swaps_teams -q`; confirm it fails under the old three-argument API.
- [ ] Implement:

```python
def mirrored(match: MatchData) -> MatchData:
    slots = {name: values.copy() for name, values in match.slots.items()}
    slots["team"] = (1 - slots["team"]).astype(match.slots["team"].dtype)
    labels = {name: swap_labels(values) for name, values in match.labels.items()}
    return MatchData(ft.swap_sides(match.table, length=match.pitch[0]), slots, labels, match.pitch, match.fps)
```

- [ ] Extend the test to assert `mirrored(mirrored(match))` restores the original table, every slot array, and labels.
- [ ] Run `python -m pytest tests/test_sportec_bundles.py::test_common_mirror_keeps_slot_geometry_and_swaps_teams -q`; confirm it passes.

### Task 3: Prove the P3+ integration and document the boundary

**Files:**

- Modify: `tests/test_common_sportec.py`
- Modify: `docs/common/COMMON_LAYER.md`

**Interfaces:**

- `classification_inputs` returns Schema 4 fields plus structured player and ball arrays through the shared API.

- [ ] Add a test that builds `MatchData` from the existing model-input test fixture and passes `InputPolicy(schema=mi.SCHEMA_PHASE, temporal_mode=ci.TemporalMode.OFFLINE_ACAUSAL)` to `sportec.classification_inputs`. Assert the returned columns equal `mi.schema_columns(mi.SCHEMA_PHASE)`, player rows/slots equal `match.slots["x"].shape`, and ball rows equal the table length.
- [ ] Run `python -m pytest tests/test_common_sportec.py -q`; confirm the new integration test fails before its implementation is available.
- [ ] Update `docs/common/COMMON_LAYER.md` to require P3-and-later branches to load `MatchData` and use `classification_inputs`, choose an `InputPolicy`, and use `mirrored(MatchData)` for training augmentation.
- [ ] Run `python -m pytest tests/test_common_sportec.py tests/test_sportec_bundles.py tests/test_model_inputs.py tests/test_classification_inputs.py -q`.
- [ ] Run `git diff --check`, inspect `git status --short`, and commit the implementation with `git commit -m "feat: share complete model input contract"`.
