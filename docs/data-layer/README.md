# Data layer: start here

This folder explains the shared data layer (loader, frame table, optional broadcast fills, run manifests, slice report) so
teammates, and the Claude sessions they brief, can use it without re-deriving anything. It contains no match data.

| file | read it when |
|---|---|
| [PIPELINE.md](PIPELINE.md) | you need to run it, find a module, or know what is cached and why |
| [SCHEMA.md](SCHEMA.md) | you need a column's meaning, unit or reliability (generated, always current) |
| [FINDINGS.md](FINDINGS.md) | you want the measured facts about the data (numbers, how they were checked) |
| [RISKS.md](RISKS.md) | before relying on any column or fill; each risk has evidence and a mitigation |
| [BROADCAST.md](BROADCAST.md) | you consider using the broadcast tracking (alignment, tiers, why fills are optional) |
| [VALIDATION.md](VALIDATION.md) | you evaluate a model: what is allowed, what proxies exist, how to find what hurts spellF1 |
| [TEAM_BRIEFS.md](TEAM_BRIEFS.md) | you own a module; per-role inputs, cautions and a prompt to give Claude |

## Ten-second summary
- One cached table, one row per tactical frame (199,319), built from the licensed bundle. Modules call `frame_table.load()`, never the raw ATD.
- Baseline = nearest player to the ball + majority vote (published floor; our build reproduces it byte for byte).
- The ball is missing on 41% of graded frames, mostly in long runs, so ball-free evidence (team shape) matters; the broadcast can fill only about 10% of gaps and did not improve spellF1 in testing.
- Nothing here has been shown to improve the ranking metric yet. Treat every added column as a candidate input to be ablated, not as a proven gain.

## Rules that bind every change (RULES.md)
No hand labels or hand corrections on the provided match; no tuning on the public five minutes; no constants tied to specific frames of this match;
derived frame-level data (the cache, outputs, manifests) stays out of the public repo (`cache/` and `outputs/` are gitignored); every method must be regenerable by one command and written up in the technical document.

## Giving this to Claude
Paste the relevant section of [TEAM_BRIEFS.md](TEAM_BRIEFS.md) plus `SCHEMA.md` and `RISKS.md`. Tell it to run `python scripts/audit_data.py` first and to quote numbers from there, not from memory.
