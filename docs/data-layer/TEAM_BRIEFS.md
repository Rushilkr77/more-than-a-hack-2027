# Per-role briefs

Each section lists what to read from the table, cautions, and a prompt you can paste to Claude together with `SCHEMA.md` and `RISKS.md`. Every brief adds: *"Run `python scripts/audit_data.py` first. Quote numbers from it, not from memory. Follow RULES.md: no hand labels/corrections on the provided match, no tuning on the public five minutes, no constants tied to specific frames."*

## Pipeline and integration (Rushil)
Owns: loader, frame table, caches, manifests, build command. Watch: schema changes bump the code fingerprint and force rebuilds; keep columns additive; keep `baseline.json` byte-identical (`cmp outputs/baseline_smooth50.csv evaluation/results/baseline_smooth50.csv`).
> Prompt: "You maintain the shared frame table. Before changing a column, list which modules read it (grep `src/`, `scripts/`), keep existing values unchanged, update SCHEMA via `scripts/render_schema_doc.py`, and prove the default baseline CSV is still byte-identical."

## Ball evidence (Priyam)
Read: `ball_visible`, `ball_x_m/ball_y_m`, `home_ball_dist_m/away_ball_dist_m`, `nearest_team`, `ball_margin_m` (confidence), `ball_home_adv_m`, optional fills via `ball_filled`. Cautions: nearest player is a defender-biased proxy (RISKS 1); margin <1 m is contested (17% of frames); ball speed from the ATD is unusable; jumps >40 m/s are tracker teleports; out-of-bounds ball is not D. Ball gaps are long, so do not interpolate.
> Prompt: "Build ball-conditioned H/A evidence with a confidence from `ball_margin_m`. Output the evidence contract columns. Do not use `ball_speed_atd_ms`. Test with burst masking and report H and A recall separately."

## Player-only ownership (Kalhan)
Read: `{home,away}_dline_m, _cen_m, _fwd_m, _line_gap_m, _shape_conf`, `home_attacks_right`, `tag_poss_hint`, `team_style()`. Cautions: shape tracks phase of play (validated vs Smart Tagging), but does not predict `nearest_team`, so do not train against `nearest_team`; `tag_poss_hint` is an unmeasured weak label; away data is noisier (goalkeeper area empty 21.7% vs 15.8%); train on both orderings with `swap_sides()`; do not use team names; no formation labels exist.
> Prompt: "Build a ball-free ownership module from the shape columns, weighting by `*_shape_conf`. Validate on other matches or held-out stretches and against Smart Tagging phases, never against `nearest_team`. Show home/away symmetry with a swap test."

## Dead-ball and restarts (Raghav)
Read: `tag_setpiece`, `tag_setpiece_team`, `pre_kickoff`, `ball_visible`, shape columns. Cautions: a missing ball is NOT dead ball; every throw-in has Team N/A, so the restart team must be inferred; out-of-bounds ball is a weak cue only (24% near restarts vs 5.8% chance); set-piece windows do not show lower player speed (1.94 vs 1.69 m/s); goal phases lose the ball.
> Prompt: "Produce `score_D` with reasons, separate from H/A. Never equate ball missing with D. Report precision against Smart Tagging restarts and state its limits."

## Decoder and evaluation (Muskan)
Read: evidence contract columns, `weight`, `anchor`, `ball_source`. Cautions: `tie="home"` exists only to keep the baseline byte-identical; use `previous` for new configs; spells are global, so use manifests plus `slice_report.py` to attribute changes; the public five minutes cannot rank configurations; the window that wins there loses on the full match.
> Prompt: "Compare decoders on whole validation matches. For each run keep the manifest, report spellF1 first then macroF1, H/A recall and home-share error, and run the burst-mask test."

## Anyone using the broadcast
Read BROADCAST.md first. Fills are optional and did not improve spellF1. Any config using them must carry `"uses_broadcast": true`.
