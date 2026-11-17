# BlrBoys: technical document

More Than a Hack 2027 · possession labelling on the BAR-ATH tactical video (199,319 frames)

## 1. Approach

We label each frame H, A or D with an algorithm that has never seen BAR-ATH labels. Four frame-classifiers each
output P(H), P(A), P(D) per frame. We average the four probability tracks and decode once with a 3-state Viterbi
smoother. The final file contains no `X`.

| member | model | notes |
|---|---|---|
| P2 | gradient-boosted trees | 282 features: per-frame tracking features plus past/next 1, 5 and 15 s mean, std and difference. 3-seed ensemble. |
| P4-30 | bidirectional GRU | 30 s windows, hidden 64 x 2 layers, 2 extra damaged copies of each training match. 3-seed ensemble. |
| P4-20 | bidirectional GRU | identical to P4-30 with 20 s windows. 3-seed ensemble. |
| P3 | dilated temporal convolutional network (TCN) | long-context windows around each frame, one run (output folder `p3_10x10_501`). |

**Blend:** equal arithmetic mean of the four probability arrays, then one Viterbi smoother with a single switch
penalty of 20 nats (the cost of one label change). The smoother never crosses halftime. Penalty 20 was tuned on
Sportec validation only.

**Why this and not one model.** On Sportec validation the members score 0.655 to 0.710 spellF1 after smoothing; the
blend scores 0.728, above every member. Blending probabilities before decoding keeps one smoother for everything, so
the comparison between approaches stays fair. We state plainly that the gain of the 4-member blend over the 3-member
blend (P2 + P4-30 + P4-20, 0.727) is within noise: its chunk-bootstrap interval crosses zero. We chose it as the best
measured point estimate, not as a confirmed improvement.

## 2. Assets used and dropped

| asset | use |
|---|---|
| Tactical ATD tracking (players, ball) | the only model input at inference, converted to a per-frame table and per-slot arrays |
| Homography | used to build the frame table |
| Organiser Smart Tagging | used only inside the smoother as set-piece and phase anchors in earlier configurations; **not** a model input |
| Our own generated Smart-Tagging-style tags (`src/features/tagger.py`) | one algorithm applied identically to training matches and BAR-ATH. Used by P3; **dropped** (`gtag_*` removed) for P2 and P4, where it gave no gain that held up |
| Broadcast video / ATD | **tried and dropped.** Broadcast-filled ball evidence was tested against the baseline and not validated to help spellF1 |
| BAR-ATH ground truth (public 5 min) | **report only.** Never used to train, select a weight, penalty, seed or member. It can only veto a collapse |

No video frames, no hand labels and no human-produced data about BAR-ATH were used.

## 3. Training data

All models are trained and tuned on **7 external matches** from the Sportec/DFL "integrated dataset of spatiotemporal
and event data in elite soccer" (figshare 28196177, licence CC BY): Bundesliga and 2. Bundesliga, 2022/23. File list
and MD5s: `data/sportec/manifest.txt`.

- **Who labelled it:** the dataset's own event and possession annotations (`BallPossession`, `BallStatus` and event
  data) are supplied by DFL. We labelled nothing by hand.
- **Raw labels (train):** `D` where `BallStatus = 0`, otherwise `H` or `A` from `BallPossession`.
- **Adjusted labels (tuning and scoring only):** a live possession spell shorter than 2 s with no control event
  (pass, cross, shot, ball claim) by that team within +/-0.5 s is treated as a touch, so possession stays with the
  previous team. The 2 s and 0.5 s constants are rule-of-thumb values, not fitted to anything.
- **Degradation:** each match is converted to BAR-ATH's tracking format and degraded by an algorithm so it resembles
  automatic, uncorrected tracking: ball gaps (35 to 50 % hidden, lognormal lengths), tracklet breaks and identity
  swaps, ghost duplicate tracks, speed blanks at tracklet starts, and ball teleports. Each mechanism has a broad
  range per match and a fixed seed (`scripts/build_sportec_bundles.py`). BAR-ATH's own visibility mask and labels are
  never read; only organiser-published headline figures set the range centres.
- **Split** (`configs/sportec_split.json`): 4 train (J03WN1, J03WOH, J03WPY, J03WR9), 2 validation (J03WMX, J03WOY),
  1 test (J03WQQ). Models also train on home/away-mirrored copies, and P4 on two extra damaged copies per match.
- **BAR-ATH** is used for inference only.

## 4. Possession rules implemented

The models learn the rules from Sportec's possession labels; the decoder enforces them.

- **Ball in play:** exactly one of H or A while play is live; `D` when the ball is dead (stoppages, restarts).
  Dead-ball status is predicted from ball and player motion (ball range, player slow share, distance to the
  touchline).
- **Loose or contested ball:** possession stays with the last team that had control. The smoother's switch penalty
  implements this: a change needs sustained evidence, so one touch, a deflection or an aerial duel does not flip the
  label.
- **Challenges:** a change of possession needs controlled possession by the opponent. The adjusted training/scoring
  labels remove sub-2 s spells with no control event.
- **Restarts:** after a stoppage play restarts with a clear possessor; the model switches `D` back to the team with
  the ball. Restarts are weak where the ball is hidden at the restart (about 39 % of Sportec restarts).
- **Halftime:** the smoother never crosses a period change.
- **Abstention:** no `X` is written; every frame gets a label.

## 5. Limitations

- **Short possessions.** On Sportec validation (629 true possessions) possessions under 3 s are found only about 33 %
  of the time. Of the 101 possessions the final output swallows, the smoother erased about 16 % and the models missed
  or were unsure about about 84 %. The model, not the smoother, is the bottleneck.
- **Ball visibility.** Possessions without a visible ball are detected about 50 % of the time, against about 75 %
  with the ball visible.
- **Merging.** The smoother merges adjacent possessions: on the BAR-ATH public window the blend outputs 10 spells
  against 17 true.
- **Weak evidence of improvement on the headline pick.** Validation was reused for roughly 30 tuning decisions, so its
  scores are optimistic. The Sportec test match was scored once and BAR-ATH has only 17 possessions in its public
  window (spellF1 interval roughly 0.4 to 0.8), so gaps below about 0.1 there are not significant.
- **Sim-to-real gap.** Our degradation is a model of BAR-ATH's tracking. BAR-ATH has larger duplicate-track bursts
  than we simulate.

With more time: a dedicated change-point head for short possessions, a better ball-gap model, and an independent
validation match for selection.

### Results

spellF1 / macroF1, switch penalty 20.

| | Sportec validation | BAR-ATH public 5 min (report only) |
|---|---|---|
| P2 | 0.668 / 0.899 | 0.736 / 0.818 |
| P4-30 | 0.710 / 0.906 | 0.675 / 0.803 |
| P4-20 | 0.700 / 0.906 | 0.629 / 0.890 |
| P3 | 0.655 / 0.886 | 0.575 / 0.870 |
| **Submitted 4-member blend** | **0.728 / 0.911** | 0.637 / 0.905 |
| Organiser baseline smooth50 | 0.219 / 0.449 | 0.326 / 0.621 |

## 6. Reproduction

### Environment
Python 3.12 (3.10+ needed for scikit-learn 1.7). Pinned: scikit-learn 1.7.2, torch 2.14.1 for P2/P4, torch 2.15.0 and
pytorch-tcn 1.2.3 for P3.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-team.txt     # P2, P4, data layer, scorer
pip install -r requirements-p3.txt       # P3 TCN (use a second venv if the torch pins clash)
```

### Data
1. **BAR-ATH match bundle (organiser, gated, not in the repository).** Enrol and download via the link in `DATA.md`.
   Unpack the tactical files into `data/tactical/` keeping their names (`BAR-ATH_tactical_atd.txt`, `_atd.xml`,
   `_homography.csv`, `_smart_tagging.xml`), or set `MTAH_DATA_DIR`. It is used for inference only.
2. **Sportec/DFL training matches (external, CC BY).** Dataset: "An integrated dataset of spatiotemporal and event data
   in elite soccer", <https://doi.org/10.6084/m9.figshare.28196177> (figshare item 28196177). The 21 files used (match
   information, raw events, observed positions for 7 matches) are listed with direct URLs and MD5s in
   `data/sportec/manifest.txt`:
   ```bash
   mkdir -p data/sportec
   while read -r url name md5; do curl -L -o "data/sportec/$name" "$url"; done < data/sportec/manifest.txt
   ```
   Check the downloads against the MD5 column (`md5 -r` on macOS, `md5sum` on Linux). The manifest ships in the
   repository; the downloaded XML files do not (`data/` is git-ignored).
3. **Build the degraded training bundles and labels** (deterministic, fixed seed per match):
   ```bash
   python scripts/build_sportec_bundles.py          # writes data/sportec_bundles/ and data/sportec_labels/
   ```
   Split: `configs/sportec_split.json`.

### Train the four members and export BAR-ATH probabilities
```bash
# P2 (final form is the script default: 3-seed ensemble, no gtag_*, 282 features)
python scripts/train_p2.py --ensemble --export                                     # -> outputs/p2/h_sd_win_1_5_15_ens

# P4 GRU, 30 s and 20 s windows (needs the damaged training copies first)
python scripts/build_p4_aug.py --copies 2
for W in 30 20; do
  python scripts/train_p4.py --window $W --seed 0 --seed 1 --seed 2
  python scripts/train_p4.py --window $W --ensemble --export                       # -> outputs/p4/04_g0_w${W}_s5_aug2_ens
done

# P3 TCN (config sets the output folder outputs/p3_10x10_501 and the seed)
python scripts/train_p3.py --config configs/p3_tcn_10x10.json --schema 04_full_phase_context
python scripts/export_p3_bar_ath.py                                                # -> .../schema_04/proba_BAR-ATH.npy
```
Each member folder then holds `proba_<match>.npy` for the validation matches and `proba_BAR-ATH.npy`.

### Produce and check the submission
```bash
python scripts/make_submission.py --out BlrBoys.csv       # equal-mean blend, one Viterbi smoother, penalty 20
python evaluation/validate_submission.py BlrBoys.csv      # 199,319 rows, labels H/A/D
python scripts/verify_best_model.py                         # checks member hashes + selected validation score
```
`make_submission.py` reads the member list and penalty from `configs/best_model_selection.json`. The blend and
smoother are deterministic: the same member probability files always produce the same CSV (we regenerated it twice and
the files were byte-identical).

### What is and is not byte-reproducible
Training uses fixed seeds and deterministic algorithms, but floating-point results can differ across machines, Python
and library builds. The submitted file was built from probabilities that reproduce the selected Sportec validation
score (0.7283 against the manifest's 0.7282); `verify_best_model.py` compares exact SHA-256 hashes and will report a
mismatch for member files trained on a different machine, even when scores agree to four decimals. The P3 member
(`p3_10x10_501`) matches the manifest hashes exactly; P2, P4-30 and P4-20 were retrained locally and do not.
