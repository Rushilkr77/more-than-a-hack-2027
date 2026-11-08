# Technical document notes

SUBMISSION.md §3 asks for a short technical document (`<team_name>.pdf` or `.md`) that explains the method well
enough for a reader to understand what we built and why (RULES.md, condition 2). This file collects method
statements as decisions are made, to compile into that document. Add an entry whenever a method decision lands.

## Training data
P2–P5 are trained and tuned only on 7 external matches: the Sportec/DFL "integrated dataset of spatiotemporal and
event data in elite soccer" (figshare 28196177, CC BY), Bundesliga and 2. Bundesliga 2022/23. Each match is converted
to BAR-ATH's tracking format and degraded by an algorithm to resemble automatic, uncorrected tracking. The
degradation covers ball gaps, tracklet breaks, ghost duplicates and speed blanks (`scripts/build_sportec_bundles.py`;
one fixed seed per match, so it is reproducible). The provided match (BAR-ATH) is used for inference only: no model,
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
