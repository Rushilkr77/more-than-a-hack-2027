"""Frozen weights -> BAR-ATH probabilities for the four blend members. No training, no labels.

    python scripts/predict_bar_ath.py            # reads weights/, writes outputs/<member>/proba_BAR-ATH.npy
    python scripts/make_submission.py --out BlrBoys.csv

weights/p2/models.joblib            P2 trees, seeds 0-2          -> outputs/p2/h_sd_win_1_5_15_ens
weights/p4_w{30,20}/seed{0,1,2}.pt  P4 GRUs                      -> outputs/p4/04_g0_w{30,20}_s5_aug2_ens
weights/p3/artifact.pt              P3 TCN (configs/p3_tcn_10x10.json) -> outputs/p3_10x10_501/schema_04
"""
import os
import sys

import joblib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]

import export_p3_bar_ath  # noqa: E402
import train_p2  # noqa: E402
import train_p4  # noqa: E402
from src.features import classification_inputs as ci  # noqa: E402
from src.features import frame_table as ft  # noqa: E402
from src.models import p2_gbt, p4_gru  # noqa: E402

W = os.path.join(ROOT, "weights")


def save(member, proba):
    path = os.path.join(ROOT, member, "proba_BAR-ATH.npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, proba)
    print("wrote", member, proba.shape)


def pitch():
    from xml.etree import ElementTree as ET
    root = ET.parse(train_p4.BAR_ATH_META).getroot()
    return float(root.findtext(".//FieldSize/Width")), float(root.findtext(".//FieldSize/Height"))


def p2():
    models = joblib.load(os.path.join(W, "p2", "models.joblib"))
    inputs = ci.make_classification_inputs(ft.load(), ft.load_slots(), pitch(), train_p2.P2_POLICY, fps=25)
    X = train_p2.features(inputs, train_p2.FINAL, 25)
    save("outputs/p2/h_sd_win_1_5_15_ens", np.mean([p2_gbt.predict_proba(m, X) for m in models], axis=0))


def p4(window):
    table = ft.load()
    inputs = ci.make_classification_inputs(table, ft.load_slots(), pitch(), train_p4.policy("04"), fps=25)
    X = train_p4.feats(inputs, gtag=False)
    probas = []
    for seed in range(3):
        saved = torch.load(os.path.join(W, "p4_w%d" % window, "seed%d.pt" % seed), weights_only=False)
        scaler, meta = saved["scaler"], saved["meta"]
        model = p4_gru.GRU(len(scaler["columns"]) + len(scaler["masked"]), meta["hidden"], meta["layers"])
        model.load_state_dict(saved["state_dict"])
        step = meta["step"]
        probas.append(p4_gru.predict_proba(model.eval(), p4_gru.transform(X, scaler), inputs.period,
                                           window * 25 // step, step))
    save("outputs/p4/04_g0_w%d_s5_aug2_ens" % window, np.mean(probas, axis=0))


def p3():
    folder = os.path.join(ROOT, "outputs", "p3_10x10_501", "schema_04")
    os.makedirs(folder, exist_ok=True)
    src = os.path.join(W, "p3", "artifact.pt")
    dst = os.path.join(folder, "artifact.pt")
    if os.path.abspath(src) != os.path.abspath(dst):
        import shutil
        shutil.copyfile(src, dst)
    export_p3_bar_ath.export(os.path.join(ROOT, "configs", "p3_tcn_10x10.json"), folder)


if __name__ == "__main__":
    p2()
    p4(30)
    p4(20)
    p3()
