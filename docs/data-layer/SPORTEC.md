# Sportec training data: damaged, BAR-ATH-shaped bundles

P2–P5 train and tune on 7 external Sportec (DFL) matches. BAR-ATH is used for inference only (organiser email,
5 Nov). The Sportec tracking is clean, so we convert each match into **exactly the raw format BAR-ATH ships** and
damage it the way an automatic tracker would. The frame table and every input method then run unchanged on both.

```
BAR-ATH   data/tactical/BAR-ATH_tactical_atd.{txt,xml}                ─┐
                                                                         ├─► build_table ─► frame table ─► input methods ─► P2–P5
Sportec   data/sportec_bundles/<M>/tactical/<M>_tactical_atd.{txt,xml} ─┘
```

## Commands
```bash
python scripts/build_sportec_bundles.py     # ~2 min, all 7 matches; raw XMLs in data/sportec/ (figshare 28196177)
python scripts/diagnostics_round1.py        # risk-register markers with a FLAG row
```
```python
from src.features import frame_table as ft
table, slots = ft.bundle_table("data/sportec_bundles/J03WMX")   # same 62 columns as ft.load() on BAR-ATH
X = table[ft.model_columns()]                                   # model inputs: no tag_*, hom_*, graded, frame
```
## Contract for the final input methods (Priyam)
The input layer turns a frame table into model inputs (PR 2 schemas plus the dead-ball block). One function, called
identically for training and inference:

```python
import json
from src.features import frame_table as ft

def make_inputs(table, pitch):            # table: one match's frame table; pitch: (length, width) metres
    ...                                   # derive PR 2 schema 01-03 fields from ft.model_columns() + window features
    return X                              # DataFrame, one row per frame, same columns for every match

split = json.load(open("configs/sportec_split.json"))
for m in split["train"]:
    table, _ = ft.bundle_table("data/sportec_bundles/" + m)
    pitch = ft.bundle_args("data/sportec_bundles/{0}/tactical/{0}_tactical_atd.xml".format(m))["pitch"]
    X = make_inputs(table, pitch)
    y = pd.read_csv("data/sportec_labels/{}/labels_raw.csv".format(m))["label"]   # the TRAINING LOOP reads labels, not make_inputs
# inference: X = make_inputs(ft.load(), GAMES["BAR-ATH"]["pitch"])
```

Rules, each tied to a risk-register row:
1. **Same function, both datasets.** No `if sportec:` branches and no BAR-ATH paths, `GAMES` constants or team names
   inside `make_inputs`. Pitch is an argument (B3/B4); for mirroring call `ft.swap_sides(table, length=pitch[0])`.
2. **Start from `ft.model_columns()`.** It already drops `tag_*`, `hom_*`, `graded`, `pre_kickoff`, `frame` and
   `ball_speed_atd_ms` (B5). PR 2 schemas 01–03 only; schema 04 (Smart Tagging) is out (Kalhan's PRs below).
3. **Never read `data/sportec_labels/`** inside input code (A2). `meta.json` damage params are never features.
4. **Windows stop at half boundaries.** Rolling or lagged features must not cross the `period` change. Sportec halves
   sit back to back with no halftime rows, so a window that crosses mixes two halves (see `_rolling` in
   `frame_table.py`).
5. **NaN stays NaN.** A missing ball or shape is information; keep the `*_conf` / `ball_visible` columns beside the
   values (PR 2 rule: never fill with 0).
6. **Parity check before training:** `make_inputs` on BAR-ATH and on one bundle must give identical column names and
   dtypes, and on Sportec no column may be constant. Then run `python scripts/diagnostics_round1.py`: every FLAG must
   be `ok`.
7. **Labels line up by row:** row i of the table is frame i of `labels_*.csv`. Train on `labels_raw`, tune the smoother
   on `labels_adjusted`, and score on the validation matches only. The test match is scored once.
8. **After the first round** write predictions as `<M>.csv` (`frame,label`) and run
   `diagnostics_round1.py --pred <smoothed> --model-only <raw>`. Act on the A6, B2, D1, D3 and D7 flags as the
   table below says.

## From bundle to P2–P5: the end-to-end flow
```
data/sportec_bundles/<M>  ──ft.bundle_table──►  frame table (62 cols, 1 row / frame)
                                                    │
                          Priyam: make_inputs(table, pitch)  ← same call on BAR-ATH (ft.load())
                                                    ▼
                          X: [frames × ~60 input columns], NaN kept + *_conf / ball_visible beside each block
                                                    │
                  ┌──────────────────┬──────────────┴─────┬───────────────────┐
                  ▼                  ▼                    ▼                   ▼
            P2 XGBoost          P3 CNN/TCN            P4 GRU           P5 Transformer
                  └──────────────────┴─────────┬──────────┴───────────────────┘
                                               ▼
                         per frame P(H), P(A), P(D) + confidence
                                               ▼
                   shared smoother (tuned on labels_adjusted, validate matches)
                                               ▼
                              frame,label  →  spellF1 / macroF1
```

**Step 1. Priyam converts (one function, every approach reads its output).** `make_inputs(table, pitch)` returns `X`
with one row per frame. It contains:

| block | source | examples |
|---|---|---|
| team shape and quality (~18) | PR 2 schema 01, from `ft.model_columns()` | `*_dline_m`, `*_cen_m`, `*_fwd_m`, `*_width_m`, `*_shape_conf` |
| shape over time (~12) | PR 2 schema 02, computed in `make_inputs` | `*_cen_change_3s`, `*_dline_fwd_max_cross_corr_5s` |
| ball and ball gaps (~15) | PR 2 schema 03 | `ball_visible`, `ball_home_adv_m`, `*_ball_dist_m`, `seconds_since_ball_seen`, `gap_length_s` |
| players around the ball (~8) | PR 2 schema 03 | `*_local_support_count_5m`, `*_tracks_ahead_ball_count` |
| dead-ball signals (~16) | frame table dead-ball block | `ball_range_*_1s_m`, `pl_slow_share`, `ball_dist_touchline_m`, `dead_ball_conf` |
| context (1) | frame table | `home_attacks_right` |

Out of `X`: Smart Tagging (`tag_*`, schema 04, smoother only), homography, raw slot positions, labels, `meta.json`.

**Step 2. Each approach reads `X` in its own shape.** The split is the same for all four: train 4, validate 2, test 1.

| | P2 XGBoost | P3 CNN/TCN | P4 GRU | P5 Transformer |
|---|---|---|---|---|
| sample | one frame = one row of `X` plus window summaries (mean/min/max over the past and next 5–10 s) | a whole half as a sequence `[frames, features]`; convolutions slide along time | a 5–10 s window `[125–250 frames, features]`, step by step | the same 5–10 s window, attention across it |
| target | label of that frame | label of every frame | label of the window's centre frame (or every frame) | same as P4 |
| NaN | native (XGBoost learns a branch for missing) | fill 0 **plus** a 0/1 missing mask channel per block | same as P3 | same as P3, plus an attention mask |
| scaling | none needed | mean/std from the **train** matches only, frozen and reused on BAR-ATH | same | same |
| windows | never across a `period` change | pad at half edges, never join halves | same | same |
| mirror augmentation | optional `ft.swap_sides(table, length=pitch[0])` with H↔A swapped in `y` | same | same | same |

All four output P(H), P(A), P(D) per frame plus a confidence. Nothing from BAR-ATH is used to fit or scale.

**Step 3. Train, tune, score.**
- **Train** (J03WN1, J03WOH, J03WPY, J03WR9): fit on `labels_raw`.
- **Validate** (J03WMX, J03WOY): early stopping and hyperparameters on the model output; the shared smoother is tuned
  on `labels_adjusted`. Rank configurations by spellF1, then macroF1, under the keep rule.
- **Test** (J03WQQ): one run at the end, with no decisions made from it.
- **Diagnostics:** `diagnostics_round1.py --pred <smoothed> --model-only <raw>` after round 1 (A6, B2, D1, D3, D7).
- **BAR-ATH:** frozen `make_inputs` + frozen model + frozen smoother → `evaluation/validate_submission.py` → submit.
  `--bar-ath-pred` checks E1 (pre-kickoff frames must be D).

## What is on disk
| path | contents | who reads it |
|---|---|---|
| `data/sportec/` | raw figshare XMLs, untouched (source of truth) | the builder only |
| `data/sportec_bundles/<M>/tactical/` | `<M>_tactical_atd.txt` / `.xml`, same line format and EPTS meta as BAR-ATH | `build_table`, input methods |
| `data/sportec_labels/<M>/labels_raw.csv` | `frame,label`, H/A/D from DFL `BallStatus` / `BallPossession` (1 = home) | training loop |
| `data/sportec_labels/<M>/labels_adjusted.csv` | same, organiser "controlled possession" grain | smoother tuning and scoring |
| `data/sportec_labels/<M>/meta.json` | seed, drawn damage params, unit check | diagnostics only, never features |
| `configs/sportec_split.json` | train 4 / validate 2 / test 1, the same for P2–P5 | everyone |

All `data/*` is git-ignored. There is one damaged version per match (fixed seed); re-running gives identical bytes.

## Kept vs dropped Sportec fields
Rule: a field reaches the bundle only if BAR-ATH has it.

| Sportec field | decision |
|---|---|
| player / ball X, Y | kept, centred metres → 0–1 normalised (corner origin, like the ATD) |
| speed `S` | kept; DFL `S` is **km/h** (checked: ÷3.6 matches displacement speed, ratio ≈ 0.99), written as m/s like BAR-ATH (BAR-ATH's meta says "kph" but its values are m/s; checked, corr 0.98) |
| ball `Z`, acceleration `A`, distance `D`, referee | dropped. `Z` is used only to place ball gaps |
| PersonId, names, shirt numbers, positions, starter flags, club ids | dropped. Slots are anonymous "Track n" with `ShirtNumber` = slot, team `team_home` / `team_away`, slot order shuffled |
| `BallStatus`, `BallPossession` | **labels only**, never in the bundle; never used to place damage |
| events XML | attack direction (`KickOff TeamLeft`) and the label adjustment only |
| homography, Smart Tagging, video | do not exist for Sportec; the columns come out as `hom_valid` False and `tag_*` 0; both are excluded by `model_columns()` |

## Damage (mechanisms, broad per-match ranges, `RANGES` in the builder)
BAR-ATH's own visibility mask is never read. Only organiser-published headline figures set range centres.
- **Ball gaps:** hidden share 35–50% per match (published: 41%); lognormal gap lengths with median 6–16 s (published: ~11 s).
  A share of gaps (0–60% per match) start at geometry-weighted frames: ball high (`Z` > 1.5 m), ≥3 players within
  3 m, ball in a box. The rest start uniformly. Labels never influence placement.
- **Tracklets:** occlusion bursts per player, identity breaks (mean tracklet 30–120 s), ghost duplicates (1–15 s,
  0.5–2 m offset), 21–26 slots per team; if no slot is free the observation is lost.
- **Speed blanks:** first 0.6–1.4 s of every tracklet (BAR-ATH: 92% blank in a tracklet's first second, a tracker
  warm-up), then 4–10% random.
- Position noise σ 0.05–0.2 m; ball speed field kept on 5–15% of visible frames.

## Labels
- **raw:** `D` where `BallStatus = 0`, else `H`/`A` from `BallPossession`. Trains the models.
- **adjusted:** a live possession spell shorter than 2 s with no control event (pass/cross `Play`, `ShotAtGoal`,
  `BallClaiming`) by that team within ±0.5 s is a touch or deflection, so possession stays with the previous team
  (CHALLENGE.md "controlled" rule). This takes about 450 → 300 spells per match, mean 7 → 11 s. It tunes and scores
  the smoother only. The 2 s and ±0.5 s are rule-of-thumb constants, not fitted to anything.

## Split (`configs/sportec_split.json`)
train J03WN1 J03WOH J03WPY J03WR9 · validate J03WMX J03WOY · test J03WQQ. J03WMX and J03WN1 are Bundesliga, the
rest 2. Bundesliga; one of each competition is in train and in validate. The test match is scored once at the end and drives no
decisions. It is a proxy: the score that ranks us is BAR-ATH's private one.

## Risk register (what `diagnostics_round1.py` checks)
Compare BAR-ATH **inputs** only to find mechanism bugs (units, origin, tracklet model). Never tune a number until
Sportec matches BAR-ATH.

| id | risk | marker | status after build |
|---|---|---|---|
| A2 | a column is a label proxy | max mutual information of any model column with H/A on live frames; flag > 0.6 bits | top is team width, 0.10–0.28 bits (real football: the team in possession spreads wide); `nearest_team` 0.07–0.13 |
| A3 | gap placement tracks the label | hidden share per H/A/D vs overall; flag skew > 0.08 | ≤ 0.06. Gap bias was capped at 0.6 after J03WOY at 0.75 skewed (dead ball hidden less) |
| A4 | clean ball used for any input | bundles hold only damaged data; no clean-ball file | by construction |
| A6 | hidden frames score like visible ones | per-visibility team accuracy on predictions | after training |
| B1 | speed unit / blanks | median speed, blank share vs BAR-ATH graded frames | 1.30–1.46 vs 1.41 m/s; 9–13% vs 8% |
| B2 | tracklet structure | tracks per team p10 / max, shape_conf | p10 8–10 vs 9; conf 0.89–0.997 vs 0.913; max 14 vs 19 (BAR-ATH has bigger duplicate bursts: known gap). `B2_share_over_13` = how often BAR-ATH goes beyond the training range. **After training:** `B2_acc_ghost` vs `B2_acc_clean` (team accuracy on frames with >11 tracks for a team vs the rest); if ghost frames are >0.05 worse, raise `duplicate_prob` / ghost duration in `RANGES` and rebuild |
| B3/B4 | pitch 105 vs 105.3, origin | each match uses its own pitch; `swap_sides(table, length=...)` | fixed |
| B5 | constant model columns | variance on Sportec | none (tags, homography excluded) |
| B7 | attack direction | team mean x at each kick-off vs `home_attacks_right` | ok all 7 |
| B8 | ball visible / gap length | per match vs published | visible 0.50–0.63 vs 0.59; gap median 6–20 s |
| C1 | label grain | spells and mean duration: raw, adjusted, predicted | see table above |
| D1, D3, D7 | overfit to validate, damage dependence, smoother doing all the work | `--pred` / `--model-only` | after training |
| E1 | BAR-ATH's second-half wait before kickoff (2,934 graded frames, always D) has no Sportec equivalent, so the model never trained on it | `--bar-ath-pred <submission.csv>`: share of `pre_kickoff` frames predicted D; flag < 0.99 | at inference; if flagged, force D on `pre_kickoff` after the smoother (the evaluator's own rule) |

## Kalhan's PRs
- PR 2 schemas 01–03 run on these bundles as they are. Schema 04 (direct Smart Tagging) stays out of P2–P5: Sportec has no tags.
- PR 4 (synthetic Smart Tagging for Sportec) is **not** used for model inputs yet. We read Sportec with our own
  `iterparse` reader because PR 4's adapter has no speed or `Z` and indexes rows across the halftime frame-number gap.
  What would have to change before its tags could feed schema 04:
  1. Phase tags use `possession = ball_possession_team_index` (DFL `BallPossession`, i.e. the label), so the tag team
     *is* the answer (risk A1). Use `nearest_team` only.
  2. Generate tags from these damaged bundles, not the clean raw XML (A4).
  3. `SET PIECES` / `GOALS` come exactly from human events, while Metrica's are loose and often `Team=N/A` (A5). Drop
     them or make them as loose as Metrica's.
  4. The generated `Confidence` label does not exist in Metrica's tags, so it must never become a feature.
  5. `TrackingFrames.participants_by_slot` (PersonId, shirt numbers, names) and `ball_is_in_play` must not reach any
     feature (A2).
  6. Missing codes (transitions, long ball, box) must be added if schema 04 is tested.
  7. Its `frame_table.py` changes (`v1`, `TRACKING_FEATURE_COLS`) must merge cleanly with the dead-ball columns on `main`.

  Once fixed, the same tag generator runs on BAR-ATH too, and the model reads *our* generated tags there, not
  Metrica's. Keep schema 04 only if it beats schema 03 under the keep rule.
