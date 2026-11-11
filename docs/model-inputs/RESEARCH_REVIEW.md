# Research review: inputs for frame-level H/A/D classification

Date: 2026-11-11

## Conclusion

The existing 20/32/60/80-column flat schemas are a sound interpretable
baseline and ablation framework. They are not, by themselves, the strongest
final representation supported by the evidence. They discard the short
trajectories and local interactions of anonymous players that the most directly
relevant possession-estimation study found useful.

The recommended input layer is therefore **hybrid**, not a larger handcrafted
CSV:

1. the existing flat frame context for quality, ball gaps, dead-ball evidence
   and interpretable ablations;
2. anonymous player-set trajectories with masks and observation ages;
3. a separate observed-ball trajectory with the same quality contract;
4. an explicit causal/offline policy;
5. no labels, events, provider tags, identities or fitted preprocessing.

This is the best defensible starting point, not a claim of perfection. Only
match-held-out ablations can select the final representation.

## Evidence reviewed

### Direct possession-estimation evidence

**Borghesi et al. (2023), “Using Temporal Convolutional Networks to estimate
ball possession in soccer games.”** This is the closest match to our H/A/D
task. It frames the output as Dead/Home/Away, embeds each tracked-object
trajectory with a TCN, and aggregates anonymous players with a
permutation-invariant operation. Its best reported design uses two output
branches and self-attention. The inputs include position, velocity, team,
object role, outside-pitch and missing flags. Its ablations report that the
five players nearest the ball can approximately match full-player possession
accuracy, while the ball trajectory is especially informative for dead/alive
classification. It uses an acausal 64-frame window at 16 fps with the target at
frame 48—approximately 3 seconds of history and 1 second of future context.

- Paper and DOI: <https://doi.org/10.1016/j.eswa.2023.119780>
- Open institutional copy: <https://iris.polito.it/handle/11583/2976470>

**Applicability limit:** the paper selected eight games with the lowest active-
play ball missingness and interpolated ball coordinates. Our damaged Sportec
bundles have only 50.3–63.4% ball visibility and median gaps of 5.9–19.7
seconds, so its interpolation policy is not transferable. Its architecture and
set/trajectory ablations are relevant; its preprocessing is not automatically
valid here.

### Dead/alive and temporal-context evidence

**Lang et al. (2022), “Predicting the in-game status in soccer with machine
learning using spatiotemporal player tracking data.”** On 102 matches, the
study uses player position, speed and acceleration and finds information around
the target—including past and future frames—useful for dead/alive status. It
splits by match, flips second-half direction, and augments by swapping home and
away teams. It reports stronger importance close to the target, particularly
within roughly four seconds.

- Open paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9522646/>
- DOI: <https://doi.org/10.1038/s41598-022-19948-1>

**Applicability limit:** this is binary in-play/stoppage classification and uses
stable known players, so it supports the temporal and symmetry decisions but
does not prove the H/A possession representation.

### Unordered player-set evidence

Tracking slots are not semantic player identities. A classifier should not
change its answer when anonymous player columns are permuted.

- **Deep Sets** proves the central permutation-invariant set-function form and
  motivates shared per-object encoders plus symmetric pooling:
  <https://arxiv.org/abs/1703.06114>
- **Set Transformer** provides attention-based interaction modeling while
  retaining permutation invariance: <https://arxiv.org/abs/1810.00825>

These sources support a player-set representation. They do not prove that
self-attention will beat sum pooling on our seven-match dataset; both remain
ablation candidates.

### Missingness evidence

**GRU-D** treats both the observation mask and elapsed time since observation
as model inputs because missingness can be informative in time series:
<https://doi.org/10.1038/s41598-018-24271-9>.

This supports preserving NaN, masks and observation ages. It does not justify
assuming every tracker gap is football information: damage mechanisms must be
matched between Sportec and BAR-ATH, and robustness must be tested by additional
masking.

### Team-geometry evidence

Football tracking research repeatedly uses centroid, width, length, spread,
interpersonal distance and running correlation as collective tactical
variables. These findings support retaining the flat shape/dynamics baseline:

- systematic review of professional-football positional references:
  <https://pubmed.ncbi.nlm.nih.gov/35173369/>
- coordination through cross-correlation and time lag:
  <https://pubmed.ncbi.nlm.nih.gov/27079483/>
- systematic review of tactical variables and coordination methods:
  <https://pubmed.ncbi.nlm.nih.gov/33030987/>

They support the existence and interpretation of the features, not their
predictive superiority for H/A/D on this dataset.

### Evaluation and preprocessing evidence

Match frames are highly correlated. Splitting individual frames would leak
match/team/tracker characteristics. The directly relevant papers separate
matches, and `GroupKFold` formalizes non-overlapping groups:
<https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html>.

Any learned scaling, imputation, feature selection or calibration must be fit
on training matches only. Scikit-learn’s leakage guidance states this
explicitly: <https://scikit-learn.org/stable/common_pitfalls.html>.

## Repository-specific findings

- Seven Sportec matches are fixed at 4 train / 2 validation / 1 untouched test.
- Frame tables contain 70 tracking-derived fields. The four flat schemas expose
  20 / 32 / 60 / 80 model features.
- Ball visibility is 50.3–63.4%; median missing-ball gaps are 5.9–19.7 s. Long
  interpolation would fabricate evidence over many possession changes.
- Anonymous bundles contain 42–52 possible slots per match with broken and
  duplicate tracklets. Slot numbers cannot be model features, and the model
  must support variable set cardinality across matches.
- Generated `gtag_poss` agrees with raw Sportec ownership on roughly
  67.3–79.7% of frames across matches, not approximately 100%. This is evidence
  that the direct BallPossession-copy shortcut is gone, but `gtag_*` remains a
  handcrafted proxy and must be ablated.
- On J03WMX, the structured player values occupy about 123 MB and their masks/
  ages about 43 MB. Build and train one match/window stream at a time; do not
  cache all derived tensors in memory simultaneously.
- BAR-ATH raw tracking is still absent locally, so cross-dataset parity remains
  unverified.

## Recommended experiment architecture

The input layer does not choose the model, but it should support the following
evidence-backed comparison:

1. **Flat baselines:** XGBoost on schemas 01–04, with schema 04’s `gtag_*`
   block separately ablated.
2. **Structured baseline:** per-player TCN with Deep-Sets sum pooling.
3. **Primary candidate:** per-player TCN with permutation-invariant attention,
   combining the structured view with selected flat quality/dead-ball context.
4. **Output comparison:** ternary H/A/D versus shared encoder with Dead/Alive
   and conditional Home/Away heads. Do not assume the paper’s winner transfers.
5. **Temporal comparison:** causal and offline-acausal windows; the competition
   is offline, but the mode must be named and cannot be silently mixed.

The first structured window to test should be approximately 3 s history + 1 s
future, matching the closest direct study after adjusting for frame rate. It is
a starting hyperparameter, not a fixed truth.

## Required ablations and failure tests

| Question | Comparison |
|---|---|
| Do handcrafted tags help or shortcut? | schema 04 with and without all `gtag_*` |
| Does object detail beat aggregates? | best flat schema vs TCN + sum pooling |
| Does learned interaction help? | sum pooling vs set attention |
| Are nearest players sufficient? | all observed players vs nearest 5, selected dynamically without labels |
| Is future context responsible for gains? | causal vs offline-acausal |
| Does missingness generalize? | normal validation vs additional ball/player masking |
| Is the model symmetric? | home/away swap and lateral reflection invariance |
| Is slot order leaking? | random slot permutation must leave predictions unchanged |
| Are results match-specific? | grouped validation and leave-one-training-match-out sensitivity |

Selection should prioritize spellF1, then macro-F1, with separate D F1,
conditional H/A accuracy, calibration and per-match dispersion reported. A
single aggregate accuracy is insufficient for long possession spells and the
minority Dead class.

## Lightweight pre-mortem

Assume the eventual model scores well on Sportec validation and fails on
BAR-ATH. The most credible causes are:

| Likely failure | Why it is credible | Early warning | Mitigation |
|---|---|---|---|
| The model learns tracker/damage style, not football. | Only four training matches; missingness and duplicate-track patterns can identify a dataset. | Accuracy falls sharply under extra synthetic masking or leave-one-training-match-out. | Match-grouped sensitivity tests, stronger damage variation, masks retained, no test-fitted preprocessing. |
| Set attention overfits the small number of matches. | It is more flexible than sum pooling and frame samples are highly correlated. | Attention wins aggregate validation but has high per-match variance or unstable seeds. | Require a parameter-matched sum-pooling baseline and prefer the simpler model unless gains are consistent. |
| `gtag_*` becomes a proxy shortcut. | Tags are safe from direct leakage but encode handcrafted nearest-team smoothing. | Removing the tag block collapses performance; predictions mirror `gtag_poss` errors. | Whole-block ablation and robustness slices; exclude tags if gains do not transfer consistently. |
| Long ball gaps are handled as confident evidence. | Ball is missing for 37–50% of frames, often for many seconds. | Confidence stays high during long gaps or switches follow gap boundaries. | No long interpolation; use masks/ages; report performance by gap duration and calibrate confidence. |
| Offline features are accidentally described or deployed as live. | Future anchors, forward windows and tag smoothing are acausal. | A causal replay differs materially from reported validation. | Mandatory `TemporalMode`; causal mode rejects acausal flat schemas. |
| Coordinate/team asymmetry leaks match conventions. | Home frame and raw flat ball coordinates can encode orientation quirks. | Home/away swap or lateral reflection changes non-swapped probabilities. | Swap/reflection augmentation and invariance tests; ablate raw coordinates/direction flags. |

The most expensive mistake would be optimizing a sophisticated network before
BAR-ATH parity is available. The input contract is ready, but architecture
selection should remain reversible until that domain check is possible.
