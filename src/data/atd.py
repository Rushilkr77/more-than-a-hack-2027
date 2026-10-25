"""Raw tactical ATD -> per-slot arrays on the submission timeline.

Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...:<bx,by,bs>", NaN for a missing observation. frame_id counts
from 1, submissions from 0, so row = frame_id - 1 (see starter/load_atd.py).
"""

import numpy as np

from starter.baseline import team_of_each_track  # slot -> 0 home / 1 away, from the ATD metadata

HOME, AWAY = 0, 1


def load_slots(meta_path, raw_path, n_frames, pitch):
    """{x, y, s: [n_frames, n_slots] float64, ball_x, ball_y, ball_s: [n_frames], team: [n_slots]}.

    x, y are metres (normalised value * pitch length / width), s is m/s, NaN where unobserved.
    Orientation is raw: no per-half flip. Every frame 1..n_frames must be present exactly once.
    """
    team = team_of_each_track(meta_path)
    n_slots = len(team)
    length, width = pitch

    out = {k: np.full((n_frames, n_slots), np.nan) for k in "xys"}
    ball = np.full((n_frames, 3), np.nan)
    seen = np.zeros(n_frames, dtype=bool)

    with open(raw_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                raise ValueError("{}: expected 3 ':'-separated fields, got {}".format(raw_path, len(parts)))
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames or seen[row]:
                raise ValueError("{}: frame_id {} out of range or duplicated".format(raw_path, row + 1))
            seen[row] = True

            players = np.array(parts[1].replace(";", ",").split(","), dtype=np.float64)
            if players.size != n_slots * 3:
                raise ValueError("frame_id {}: {} values for {} slots".format(row + 1, players.size, n_slots))
            players = players.reshape(n_slots, 3)
            out["x"][row], out["y"][row], out["s"][row] = players[:, 0], players[:, 1], players[:, 2]
            ball[row] = np.array(parts[2].split(","), dtype=np.float64)

    if not seen.all():
        raise ValueError("{}: {} of {} frames missing".format(raw_path, int((~seen).sum()), n_frames))

    out["x"] *= length
    out["y"] *= width
    # an observation needs both coordinates
    gone = ~(np.isfinite(out["x"]) & np.isfinite(out["y"]))
    out["x"][gone] = out["y"][gone] = out["s"][gone] = np.nan
    out.update(ball_x=ball[:, 0] * length, ball_y=ball[:, 1] * width, ball_s=ball[:, 2], team=team)
    ball_gone = ~(np.isfinite(out["ball_x"]) & np.isfinite(out["ball_y"]))
    out["ball_x"][ball_gone] = out["ball_y"][ball_gone] = out["ball_s"][ball_gone] = np.nan
    return out
