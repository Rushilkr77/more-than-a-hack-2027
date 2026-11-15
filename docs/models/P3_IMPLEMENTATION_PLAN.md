# P3 implementation plan

The approved P3 submission pipeline is now implemented up to training.

```text
Sportec train bundles + raw labels
  -> shared classification-input builder
  -> exact schema 03 OR exact schema 04
  -> train-only standardisation, no extra input channels
  -> 101-frame period-safe target windows (3s past, target, 1s future)
  -> causal pytorch-tcn -> final output aligned to target frame
  -> weighted cross-entropy and validation-loss early stopping

Sportec validation
  -> probabilities -> existing shared Viterbi tuning
  -> smoothed spellF1, then macroF1 -> selected schema artifact

Held-out Sportec test / BAR-ATH
  -> selected frozen artifact only; no further fitting or selection
```

## Implemented files

| File | Responsibility |
|---|---|
| `configs/p3_tcn.json` | All temporal, architecture, and optimisation parameters. |
| `src/models/p3_tcn.py` | Exact schemas, scaler, target windows, package TCN wrapper, and safe artifacts. |
| `scripts/train_p3.py` | Runs the two fixed schema trainings, validation scoring, and selection record. |
| `requirements-p3.txt` | Isolated PyTorch and `pytorch-tcn==1.2.3` dependency set. |
| `tests/test_p3_tcn.py` | Contract tests for schemas, scaling, windows, model shape, artifacts, and selection. |

## Before invoking training

Run:

```bash
.venv/bin/python -m pytest tests/test_p3_tcn.py -v
.venv/bin/python scripts/validate_model_inputs.py
.venv/bin/python scripts/train_p3.py
```

The final command trains schema 03 and 04 sequentially and writes ignored
artifacts to `outputs/p3/selection.json` plus one directory per schema.
