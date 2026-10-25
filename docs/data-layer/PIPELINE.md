# Pipeline

```text
licensed bundle (outside git)  ->  src/data/atd.py  ->  src/features/frame_table.py  ->  cache/*.parquet, slots npz
cache  ->  src/evidence/*  (scores)  ->  src/decoder/majority.py (labels)  ->  scripts/build_submission.py  ->  outputs/<run>.csv + .manifest.json
optional: broadcast ATD -> src/features/broadcast_xcheck.py -> cache/broadcast_xcheck_v0.parquet -> src/evidence/ball_filled.py
```

## Setup and commands
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-team.txt
.venv/bin/python -m pytest -q                         # synthetic tests always; real-data tests when the bundle is present
.venv/bin/python scripts/audit_data.py                # re-runnable facts about the data (prints only)
.venv/bin/python scripts/build_submission.py          # default baseline -> outputs/baseline_smooth50.csv (+ manifest)
.venv/bin/python scripts/broadcast_xcheck.py          # broadcast confidence matrix + outputs/broadcast_fill_segments.csv
.venv/bin/python scripts/validate_fill.py --seeds 8   # burst-mask test of fills
.venv/bin/python scripts/slice_report.py --pred A.csv --gt labels.csv   # scores sliced by reliability markers
.venv/bin/python scripts/render_schema_doc.py         # regenerate SCHEMA.md (a test checks it is current)
```
Data goes in `data/tactical/` (or set `MTAH_DATA_DIR`). Broadcast files may sit in any folder under `data/broadcast/`.

## Modules
| path | job |
|---|---|
| `src/data/atd.py` | parse raw ATD to per-slot arrays in metres (`row = frame_id - 1`); fails loudly on missing/duplicate frames |
| `src/features/frame_table.py` | build/cache the table; `load()`, `load_slots()`, `swap_sides()`, `team_style()`, `shape_columns()` |
| `src/features/broadcast_xcheck.py` | align broadcast to tactical, grade tiers 1-4, cache the result |
| `src/evidence/ball_nearest.py` | baseline evidence: team of the track nearest the ball (confidence stays 0/1) |
| `src/evidence/ball_filled.py` | OPTIONAL evidence with broadcast fills; adds `weight`, `anchor`, `ball_source`, `align_conf` |
| `src/decoder/majority.py` | weighted majority vote; `min_anchor`; `tie` = "home" (baseline, byte-identical) or "previous" (neutral) |
| `scripts/build_submission.py` | evidence + decoder from a config, validates with the official checker, writes the manifest |
| `configs/*.json` | `baseline` (default), `filled_t1`, `filled_t12` (marked `uses_broadcast`, print a warning) |

## Evidence contract
`frame, score_H, score_A, score_D, confidence` (NaN score = no opinion). `ball_nearest` keeps `confidence` as 0/1. Optional columns (`weight`, `anchor`, `ball_source`, `align_conf`) are supplementary; the decoder uses `weight` if present.

## Caches invalidate themselves
`cache/` holds the frame table, per-slot arrays and the broadcast cross-check. Each records a hash of the code that built it, the build arguments and the input files; `load()` rebuilds on any mismatch (needs the raw bundle). If raw inputs are absent it warns and checks code only. Never copy a cache between machines as if it were source.

## Outputs
`outputs/<run_id>.csv` (frame,label) and `<run_id>.manifest.json`: config, code and frame-table fingerprints, output checksum, `uses_broadcast`, and the share of graded frames resting on each evidence kind (tactical ball, tier-1/2 fill, none, low margin, low shape confidence) plus the label mix. Compare manifests when a score moves.

## Add a new evidence module
Return the contract columns keyed by `frame` (scores as evidence, never final labels); register it in `EVIDENCE` in `build_submission.py`; add a config; keep `confidence` meaning stable; add a synthetic test; mark anything non-default with `uses_<source>` in the config.
