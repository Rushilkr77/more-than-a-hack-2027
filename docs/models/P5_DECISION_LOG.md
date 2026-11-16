# P5 decision log

Append new entries; do not silently rewrite an earlier decision after results
arrive. Separate repository evidence from assumptions.

## 2026-11-16 — P5-001: define completion independently of a win

- **Status:** accepted.
- **Decision:** P5 is complete when a reproducible Transformer comparison has
  passed all four gates and the keep/stop rule is recorded. Completion does not
  require P5 to beat P4.
- **Evidence:** P6 already demonstrates why a negative model result can be a
  valid deliverable when the implementation, runs, and stop decision are
  reproducible.
- **Rejected alternative:** keep extending attention until it wins. That would
  turn the reused validation set into an implicit training set.

## 2026-11-16 — P5-002: use a small local-global frame-token encoder

- **Status:** accepted for v0.
- **Decision:** use one low-capacity local temporal convolution residual plus a
  two-layer Transformer over sampled frame tokens, with fixed sinusoidal
  positions and a per-token H/A/D head.
- **Evidence:** the original Transformer establishes attention plus positional
  encoding for ordered sequences. Repository evidence says four matches
  overfit quickly and a hidden width of 32 already hurt P4, so width 64 with two
  layers is a controlled middle ground rather than an aggressive scale-up. The
  possession literature supports local temporal encoding, while attention tests
  non-local context; neither is assumed to remove match bias.
- **Assumption:** a 150-token window is long enough to expose the temporal
  dependencies P5 is intended to test.
- **Invalidated if:** Gate 1 shows the parameter budget is far above P4 or the
  sequence contract cannot reconstruct one probability row per input frame.

## 2026-11-16 — P5-003: make the first comparison architecture-only

- **Status:** accepted.
- **Decision:** match P4's schema 04 without `gtag_*`, train-only scaling plus
  missingness flags, 30-second window, step 5, two extra damaged training
  copies, three seeds, common scorer, and common smoother.
- **Evidence:** P4 measured the damaged copies and seed ensemble as material
  gains. Changing them would confound the comparison more than changing the
  recurrent encoder to self-attention.
- **Uncertainty:** the P4 augmentation helper is not on `main` yet. P5 should
  not import an unmerged branch; Gate 2 must either wait for P4 to merge or move
  the generic augmentation into a separately reviewable common change.

## 2026-11-16 — P5-004: no initial patching or bespoke attention

- **Status:** accepted.
- **Decision:** v0 uses one sampled frame per token and standard full
  self-attention. No patches, convolutions, relative positions, sparse
  attention, or pretrained model.
- **Evidence-backed inference:** patching can reduce attention cost and improve
  some forecasting tasks, but it also changes temporal resolution and the
  learning problem. At 150 tokens, cost is not yet the binding constraint.
- **Rejected alternative:** adopt a sophisticated time-series Transformer
  before establishing that standard attention helps this classification task.
- **Trade-off:** v0 may understate the best attainable Transformer score, but it
  yields a clean and falsifiable comparison.

## 2026-11-16 — P5-005: evidence threshold and stopping

- **Status:** accepted.
- **Decision:** P5 advances only when its three-seed ensemble beats P4 and the
  `keep_rule` 95% interval for validation smoothed spellF1 is wholly positive.
  Otherwise record the negative result and stop.
- **Evidence:** P4 reports neural seed spread around 0.03-0.04, a 0.025 ensemble
  gain, and optimism from repeated validation use. A point estimate is not
  enough.
- **Trade-off:** this rule may reject a genuinely useful but small improvement;
  that is preferable to selecting noise with only two validation matches.

## 2026-11-16 — P5-006: hybrid architecture is not a match-bias control

- **Status:** accepted.
- **Decision:** add the local convolution path, but judge match bias separately:
  no identity fields, whole-match grouping, per-match validation direction,
  leave-one-training-match-out sensitivity, and label-independent masking.
- **Fact:** convolution plus attention can learn more patterns than either path
  alone; it does not mathematically enforce domain invariance.
- **Evidence-backed inference:** if a gain appears on only one validation match
  or collapses when tracker missingness changes, the likely explanation is a
  match/damage shortcut rather than a general football signal.
- **Rejected alternative:** treat a hybrid model as sufficient protection from
  match bias without testing cross-match stability.

## 2026-11-16 — P5-007: shorten the fixed window for deadline throughput

- **Status:** accepted; supersedes P5-002's 150-token/30-second assumption.
- **Decision:** use a 10-second window at step 5 (50 tokens) for the three-seed
  P5 experiment.
- **Evidence:** P4's controlled comparison found 10, 30, and 60 second windows
  tied. The first 30-second P5 seed ran nine epochs successfully but was too
  slow to complete three seeds and evidence checks before the deadline.
- **Observed before stopping:** validation cross-entropy improved from 0.415 at
  epoch 0 to 0.303 at epoch 7, so the architecture was learning; the run was
  stopped for throughput, not model failure.
- **Trade-off:** this is no longer an exact window-level match to final P4, but
  it uses a P4-tested tied setting and preserves every other comparison control.

## 2026-11-16 — P5-008: permit one fixed heterogeneous blend, not weight search

- **Status:** accepted as the only recursive model change after the standalone
  P5 ensemble.
- **Decision:** if standalone P5 does not beat P4, evaluate one 50/50 arithmetic
  average of the frozen P5 ensemble probabilities and the existing P2 GBT
  probabilities, then apply the same common smoother once. Do not search blend
  weights, change either member, or add a learned stacking model.
- **Evidence:** P2's hand-built rolling statistics and P5's learned temporal
  representation are materially different error mechanisms. Prediction
  averaging is an established, simple ensemble baseline; the Deep Ensembles
  study reports stronger predictive performance and robustness from averaging
  independently trained predictors:
  <https://papers.nips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html>.
- **Uncertainty:** that paper studies neural ensembles, not this football task
  or a tree/Transformer pair. The blend is therefore a falsifiable experiment,
  not evidence that diversity must help here.
- **Bias control:** report both validation matches separately. A mean gain that
  regresses either match is not accepted as evidence against match-specific
  behavior.
- **Stop condition:** whether the fixed blend wins or loses, do not try another
  weight on the reused two-match validation set.

## 2026-11-16 — P5-009: reject P5

- **Status:** final.
- **Fact:** the P5 three-seed ensemble scored 0.675 validation spellF1 / 0.900
  macroF1. P4 records 0.710 / 0.906.
- **Fact:** the pre-declared equal P2/P5 blend improved P5 to 0.695 / 0.910,
  but remained below P4. Its keep-rule gain over P2 was +0.0057 with a 95%
  interval of [-0.0115, +0.0236], so it failed the common keep threshold.
- **Decision:** complete P5 as a negative result. Do not search blend weights,
  grow the Transformer, or touch the held-out test match.
- **Match-bias evidence:** P5 regressed P2 on one validation match; the hybrid's
  chunk gain over P2 was negative on J03WMX and positive on J03WOY. Neither
  candidate passes the required non-negative cross-match direction gate.
- **Missing evidence:** leave-one-training-match-out and extra masking retrains
  were stopped after the primary and cross-match gates failed. They are needed
  before any future attempt to revive P5, but cannot change this rejection.

## 2026-11-16 — P5-010: make the P2 dependency reproducible

- **Status:** accepted.
- **Problem found:** the checked-in P2 runner did not save the probabilities
  required by the fixed hybrid, and scikit-learn 1.6 lacks the explicit
  validation-set API used by the source.
- **Decision:** save P2 probabilities and periods, detect API support, and
  record either `explicit_grouped_validation` or `internal_train_fraction` in
  metadata. Rerun P2 and the hybrid after this correction; discard the earlier
  hybrid number.
- **Final rerun:** P2 0.685 spellF1; hybrid 0.695. The correction did not change
  the final P5 rejection.

## 2026-11-16 — P5-011: update the incumbent after live GitHub verification

- **Status:** final evidence update; no new P5 experiment.
- **Fact:** PR 16 is open and reports an equal P2 + GRU 30 s + GRU 20 s blend
  at 0.727 validation spellF1 / 0.912 macroF1 and 0.718 / 0.911 on the one-shot
  test. Its reported keep-rule interval against P4 is [+0.001, +0.038] on
  validation and [+0.008, +0.036] on test.
- **Decision:** call PR 16 the best recorded candidate and P4 the best recorded
  single model. Both remain open, with no checks or reviews reported by GitHub.
  This strengthens the P5 rejection; it does not authorize another P5 blend.
- **Caution:** PR 16 reports 24 validation candidates and a validation lower
  bound only narrowly above zero. Its one-shot test agreement is important,
  but its ignored probability artifacts are not available on this branch for
  independent reproduction.

## Open decisions

| ID | Question | When it may be decided | Required evidence |
|---|---|---|---|
| OPEN-P5-01 | Exact parameter-count band relative to P4 | Resolved: 79,747 P5 vs approximately 132,739 P4 trainable parameters at 123 inputs | Measured P5 count and direct count from P4's recorded architecture. |
| OPEN-P5-02 | Share P4's augmentation helper or wait for PR 14 | Resolved: a P5-local wrapper reuses the common deterministic damage builder and seed offsets | Same ranges and seed rule; no P4 model dependency. |
| OPEN-P5-03 | Early-stop patience or learning-rate change | Resolved: no change; training was stable | Three finite loss curves and deterministic tests. |
