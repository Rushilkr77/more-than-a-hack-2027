# Model input layer: team handoff and next steps

Date: 2026-11-11

## Short version

We are in a substantially better position than before for **data processing and
leakage control**. The seven Sportec matches can now be converted into the same
tracking contract, labels remain separate, and every model input is built by a
tested allow-list. We also added player and ball trajectories for temporal/set
models instead of forcing every model to use one flat CSV.

This does **not** yet prove that the final H/A/D model will be better. No model
comparison has been run, and BAR-ATH parity cannot be completed until its raw
tracking bundle is available locally. The new layer gives us better evidence,
safer experiments and more suitable inputs; the next ablations must prove the
performance gain.

## What changed and why

| Earlier | Now | Why it is better |
|---|---|---|
| Sportec BallPossession could influence Smart Tagging inputs while also providing the target. | Sportec labels are stored separately; generated `gtag_*` reads tracking only. | Removes the direct answer-copy leakage path. |
| Model inputs could grow from the frame table without a strict per-schema boundary. | Each flat schema has an explicit ordered allow-list. | New table columns cannot silently become model features. |
| PR 2 described four standalone CSV schemas but there was no final builder. | Schemas 01–04 are implemented as 20 / 32 / 60 / 80-feature experiments. | Preserves the original controlled comparisons and makes them reproducible. |
| The final approach was effectively a flat feature table. | The consolidated layer provides both flat context and anonymous player/ball trajectories. | Temporal/set models retain local movement and player interactions that aggregates discard. |
| Anonymous slot order could be mistaken for player identity. | Slots have no identity feature and the model contract requires permutation-invariant aggregation. | Prevents learning arbitrary tracker column order. |
| Missing ball/player values had no complete temporal quality contract. | Values remain missing and are accompanied by masks and time since observation; offline mode may also expose time until observation. | Makes missingness explicit without inventing long hidden trajectories. |
| Future-derived inputs were not clearly separated from live inputs. | Every caller must choose `causal` or `offline_acausal`; causal mode rejects incompatible schemas. | Prevents accidental claims that an offline model can run live. |
| Validation mostly checked whether files existed. | The preflight checks all seven matches, split integrity, label alignment, columns, dtypes, constants, missing features, provider-tag exclusion and structured shapes. | Fails early when the training contract drifts. |
| Design reasoning was spread across chat and PR discussion. | Research review, decision log and pre-mortem are stored with the code. | Future changes can be reviewed against their evidence and assumptions. |

## What is verified

- All seven Sportec matches build successfully, with approximately one million
  aligned frames in total.
- The fixed split remains 4 train / 2 validation / 1 untouched test.
- All four flat schemas pass across every match with identical ordered columns
  and dtypes.
- No provider `tag_*`, label, event, match identity or damage metadata is
  exposed by the input builders.
- Structured inputs support 42–52 anonymous slots depending on the match.
- Ball visibility ranges from 50.3% to 63.4%; missing values are retained with
  quality masks instead of long interpolation.
- Repository tests pass: **71 passed, 6 skipped**.
- The broader Sportec risk diagnostics previously reported `FLAG ok` for every
  match.

## Confidence assessment

These numbers are engineering confidence estimates, not statistically
calibrated probabilities.

| Area | Confidence | Reason |
|---|---:|---|
| Direct BallPossession leakage is removed from model inputs | **95% — high** | Source boundary, allow-lists, code inspection and tests agree. A future code change could still reintroduce leakage, so the preflight must remain mandatory. |
| Sportec input construction is internally correct and repeatable | **90% — high** | All seven matches pass row, schema, dtype and structured-shape validation. |
| Sportec and BAR-ATH will have input parity | **65% — medium** | Both use the same builder, but the actual BAR-ATH raw bundle has not been run through the new consolidated gate locally. |
| The hybrid inputs are more suitable than the earlier flat-only approach | **80% — medium-high** | Direct possession research supports object trajectories and permutation-invariant aggregation; our implementation preserves the simpler baseline. |
| The hybrid approach will improve held-out H/A/D performance | **60% — medium** | Plausible and research-supported, but no repository model ablation has measured it yet. Four training matches is a small number of independent games. |
| The project is ready for final BAR-ATH inference | **40% — low-medium** | Input construction is ready, but model selection, calibration, smoother tuning and cross-dataset parity remain. |

Overall: confidence in the **data/input foundation is high**; confidence in the
**eventual model gain is medium** until we run the experiments below.

## Will this help the end goal?

Probably, but for a specific reason: the new layer lets the model see short
player and ball trajectories while still keeping the safer flat baselines. The
closest published H/A/D possession work found that player trajectories were
important for ownership and ball motion was important for dead/alive status.
Our hybrid contract can test that architecture without committing to it.

The improvement is therefore not “more features must be better.” The improvement
is that we can now compare:

1. simple, interpretable flat models;
2. temporal models over raw anonymous trajectories;
3. sum pooling versus set attention;
4. ternary H/A/D versus Dead/Alive plus conditional Home/Away heads;
5. causal versus offline inference;
6. generated tags versus no generated tags.

That comparison is much closer to the actual end goal than selecting a complex
model first and adapting the data around it.

## Main concerns

### 1. Sportec-to-BAR-ATH domain shift

The biggest unresolved risk is that the model learns Sportec damage/tracker
patterns that do not match BAR-ATH. Running BAR-ATH through the same input layer
is the highest-priority gate.

### 2. Only four independent training matches

One million frames is not one million independent examples. Frames from the
same match are highly correlated. Attention-heavy models can overfit the four
training matches even when frame accuracy looks strong.

### 3. Generated tags may become a shortcut

`gtag_*` is tracking-only, so it is not the original leakage. However,
`gtag_poss` is still a handcrafted nearest-team proxy. Schema 04 must always be
compared with the entire tag block removed.

### 4. Missingness may identify the dataset

Masks are useful, but the model could learn tracker-specific gap patterns. We
need extra synthetic masking and per-gap-duration evaluation to show that the
model learned football rather than the damage recipe.

### 5. Offline context can inflate expectations

Future anchors and forward windows are valid for this offline competition, but
they cannot be described as real-time inference. Causal and offline results
must remain separate.

### 6. Raw orientation and home/away asymmetry

The original flat schemas retain some raw coordinates for compatibility. The
winning model must pass home/away swap, lateral reflection and slot-permutation
tests.

## Next steps and gates

### Gate 1 — Review and merge the input-layer PR

- Review the source boundary, four schema lists and explicit temporal policy.
- Run:

  ```bash
  python -m pytest -q
  python scripts/validate_model_inputs.py
  python scripts/diagnostics_round1.py --no-reference
  ```

- Do not include trained models, predictions or test-set decisions in this PR.

**Exit condition:** tests pass and every Sportec match reports `ok`.

### Gate 2 — BAR-ATH parity

- Place the licensed BAR-ATH tactical ATD files under `data/tactical/`.
- Build the same consolidated inputs.
- Compare ordered columns, dtypes, ranges, missingness and slot/mask shapes.
- Investigate distribution differences; do not “fix” BAR-ATH using its labels.

**Exit condition:** the same code builds both datasets, with every difference
explained by data rather than a dataset-specific branch.

### Gate 3 — Flat baselines

- Train one simple tree baseline for schemas 01, 02, 03 and 04.
- Add schema 04 without all `gtag_*`.
- Fit preprocessing on the four training matches only.
- Rank on the two validation matches by spellF1, then macro-F1.
- Keep J03WQQ untouched.

**Exit condition:** we know which flat feature blocks add consistent per-match
value and whether `gtag_*` helps or shortcuts.

### Gate 4 — Structured trajectory baselines

- Start with a small per-player TCN and sum pooling.
- Compare permutation-invariant attention with a similar parameter budget.
- Start with approximately 3 s history + 1 s future, then tune only on
  validation matches.
- Compare all players with a dynamically selected nearest-five view.

**Exit condition:** a structured model must beat the best flat baseline on both
validation matches, not only the combined score, and remain stable across seeds.

### Gate 5 — Robustness and symmetry

- Add ball/player masking beyond the normal damage level.
- Score by visible-ball and gap-duration slices.
- Test home/away swap, lateral reflection and random slot permutation.
- Compare causal and offline modes separately.

**Exit condition:** gains survive missingness and invariance tests without a
large home/away or per-match imbalance.

### Gate 6 — Freeze, test once, then infer BAR-ATH

- Freeze input policy, preprocessing, architecture and smoother.
- Score the untouched Sportec test match once.
- Make no further design choices from that result.
- Run frozen inference on BAR-ATH and validate the submission format.

## Suggested PR scope

Include:

- leak-safe flat schema builder;
- consolidated structured input contract;
- timestamp compatibility fix needed to build the data;
- preflight and unit tests;
- updated Sportec documentation;
- research review and decision log.

Exclude:

- model training code;
- selected hyperparameters;
- prediction files;
- test-match results;
- claims that attention or schema 04 is already the winner.

## Recommendation

Raise the PR as an **input-contract and leakage-prevention PR**, not as a model
performance PR. The strongest claim supported today is:

> We now have a reproducible, leakage-safe and architecture-ready input layer
> that passes all seven Sportec matches and supports controlled flat and
> trajectory-model experiments. Performance and BAR-ATH transfer remain to be
> demonstrated.

That claim is useful, accurate and reviewable.
