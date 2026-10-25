"""Render docs/data-layer/SCHEMA.md from the SCHEMA in src/features/frame_table.py (a test keeps them in sync).

    python scripts/render_schema_doc.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from src.features import frame_table as ft  # noqa: E402

DOC = os.path.join(ROOT, "docs", "data-layer", "SCHEMA.md")

HEAD = """# Frame table schema ({version})

GENERATED from `SCHEMA` in `src/features/frame_table.py` by `scripts/render_schema_doc.py`. Do not edit by hand; a test fails if this drifts.

One row per tactical-video frame (199,319 rows), `frame` 0-based, row i == frame i. Load it with `frame_table.load()`.

Conventions
- Raw position columns (`*_cx_m`, `ball_x_m`, ...) are metres on a {length} x {width} m pitch in the ATD's own orientation (NOT flipped per half).
- Shape and ball-territory columns (`*_dline_m`, `*_cen_m`, `*_fwd_m`, `*_line_gap_m`, `ball_home_adv_m`) are "metres from that team's own goal line": bigger = more advanced, same meaning in both halves, via `home_attacks_right`.
- Missing is NaN for floats; counts are 0; booleans are never null. A missing ball is `ball_visible == False`, never "dead ball".
- Reliability columns: `home_shape_conf`, `away_shape_conf` (0..1), `ball_margin_m` (bigger = more trustworthy `nearest_team`). Hints (`tag_poss_hint`) are noisy weak labels.

| column | dtype | meaning |
|---|---|---|
"""


def render():
    text = HEAD.format(version=ft.SCHEMA_VERSION, length=GAMES[GAME]["pitch"][0], width=GAMES[GAME]["pitch"][1])
    return text + "".join("| `{}` | {} | {} |\n".format(*row) for row in ft.SCHEMA)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    open(DOC, "w").write(render())
    print("wrote", os.path.relpath(DOC, ROOT))
