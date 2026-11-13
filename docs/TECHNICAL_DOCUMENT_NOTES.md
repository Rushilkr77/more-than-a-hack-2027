# Technical document notes

SUBMISSION.md §3 asks for a short technical document (`<team_name>.pdf` or `.md`) that explains the method well
enough for a reader to understand what we built and why (RULES.md, condition 2). This file collects method
statements as decisions are made, to compile into that document. Add an entry whenever a method decision lands.

## Training data
P2–P5 are trained and tuned only on 7 external matches: the Sportec/DFL "integrated dataset of spatiotemporal and
event data in elite soccer" (figshare 28196177, CC BY), Bundesliga and 2. Bundesliga 2022/23. Each match is converted
to BAR-ATH's tracking format and degraded by an algorithm to resemble automatic, uncorrected tracking. The
degradation covers ball gaps, tracklet breaks, ghost duplicates and speed blanks (`scripts/build_sportec_bundles.py`;
one fixed seed per match, so it is reproducible). Tracker
error is low-frequency with light output smoothing, as real optical trackers produce; its ranges are wide and chosen
from that mechanism, not fitted to BAR-ATH. The provided match (BAR-ATH) is used for inference only: no model,
scaler, threshold or smoother setting is fitted on it, on its video or on its ground truth. Split: 4 training, 2
validation and 1 test match (`configs/sportec_split.json`).

## Smart Tagging
We generate our own Smart-Tagging-style phase tags from tracking with one algorithm (`src/features/tagger.py`). It is
applied identically to the training matches and to BAR-ATH, so the model sees the same kind of inputs in training and
at inference. The organiser-provided Smart Tagging has no equivalent for the training matches, so it is not a model
input; it is used only in the smoother (set-piece and phase anchors). RULES.md allows this ("Improving that data, or
replacing it with your own, is entirely legitimate", provided it is produced by an algorithm). No hand labels,
ground truth or video are involved. The tagger's rules come from published phase definitions (zone rule;
post-processing per Li & Link 2026; transition window per Bauer & Anzer 2021). None of its constants is fitted to
Metrica's tags or to any label.

## P2: gradient-boosted trees
- **Model:** histogram gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`, default settings) predict
  per-frame P(H), P(A), P(D). Inputs are the frame-table columns plus, for ball, team-shape, speed and dead-ball
  columns, the past and next mean, standard deviation and next-minus-past difference over 1, 5 and 15 s windows inside
  each half (260 inputs). Trained on 4 damaged Sportec matches (every 5th frame, plus a home/away-mirrored copy), with
  early stopping on 2 validation matches.
- **Smoother:** a 3-state Viterbi decoder with one switch penalty, tuned on the validation matches against the
  organiser-grain labels. It never crosses halftime.
- **Selection:** each change was kept only if a bootstrap over 2-minute validation chunks put its spellF1 gain above 0
  (ties go to the simpler option). Tree settings, class weights, more rows, scoring-grain training labels, a two-stage
  stack and smoother alternatives (separate penalties, minimum possession length, class prior shift) all tied or lost.
  Temporal context (the windowed features) gave the gain.
- **Result:** validation spellF1 0.664, macroF1 0.899 (organiser baseline on the same matches 0.20–0.24).
- **Risks:**
  - The selection used 2 validation matches, so the score is optimistic.
  - BAR-ATH's tracker loses the ball more in calm play than our damage does, which we deliberately do not copy.
  - We deliberately did not fit the damage to BAR-ATH's numbers or its missing-frame pattern. BAR-ATH is inference
    only, and the models should learn broad tracker behaviour, not one match.
  - Details: `docs/models/P2.md`.
