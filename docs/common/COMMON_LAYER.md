# Common layer for P2–P5

Every approach is compared on the same inputs, split, smoother and scorer, so the ranking reflects the models and not
the plumbing. Everything here is shared; a model branch should only add its own model, inputs and training.

## What is shared
| piece | where | rule |
|---|---|---|
| Training data | `scripts/build_sportec_bundles.py` → `data/sportec_bundles`, `data/sportec_labels` | Same damaged bundles for all (fixed seeds). Damage is described in `docs/data-layer/SPORTEC.md`. Rebuild after pulling if it changed. |
| Split | `configs/sportec_split.json` | 4 train / 2 validate / 1 test. The test match is never loaded by training code. |
| Loading, mirroring | `src/common/sportec.py` `load`, `mirrored` | `labels_raw` trains; `labels_adjusted` tunes and scores. Train on a home/away-mirrored copy. |
| Smoother | `src/decoder/viterbi.py` `smooth` | 3-state Viterbi over P(H), P(A), P(D), one switch penalty, never crosses halftime. |
| Penalty tuning | `src/common/sportec.py` `tune_penalty`, grid `PENALTIES` | Penalty (in nats, cost of one label change) picked by mean validation spellF1 against `labels_adjusted`. Never tuned on BAR-ATH. |
| Scorer | `src/common/sportec.py` `score` (wraps `evaluation/possession_metrics.py`) | spellF1 is the headline, macroF1 the tie-break. |
| Keep rule | `src/common/keep_rule.py` | A change is kept only if its spellF1 gain has a 95% bootstrap interval above 0 over 2-minute validation chunks; ties go to the simpler option. |
| Baseline | `src/common/sportec.py` `baseline` | Organiser nearest-player vote, smooth50, not tuned. |

## Headline number
Report spellF1 and macroF1 after the **common smoother** with its penalty tuned by `tune_penalty`, next to the
model-only argmax (so the smoother's share of the score stays visible) and the baseline.

A model may also try its own smoother. Report that as an extra row, never in place of the common number, so the
approaches stay comparable. If a different smoother is better for everyone, change `src/decoder/` in a PR of its own
and re-run all approaches.

## The smoother and its penalty
Viterbi finds the label sequence that maximises the sum of log P(label) minus a penalty per label change. Cost is
measured in nats (natural-log units): a penalty of 20 asks for strong, sustained disagreement with the model before a
label flips. 0 is plain argmax; larger gives fewer, longer spells. Optional extras on `smooth` (separate H↔A and
live↔D penalties, minimum spell length) default off. They tied the single penalty in P2 (`docs/models/P2.md`), so the
default stays one penalty.

## Do not
- change the split, damage ranges, labels or scorer inside a model PR;
- tune anything on BAR-ATH or load the test match before the final run;
- add a second headline number from a private smoother.
