"""
Dumb baseline: color histogram + logistic regression  (NJ vs non-NJ).

Why this exists
---------------
Before trusting a ResNet, you want a number to beat. NJ plates have a
distinctive tan/yellow "Garden State" background; PA and many others are bluer
or whiter. So a model that looks ONLY at the distribution of colors in the crop
-- ignoring position, characters, shape, everything spatial -- should already
get real signal. If the ResNet can't beat this, something is wrong (bad split,
broken labels, a training bug).

What it does, precisely
-----------------------
1. For each crop: convert to HSV, then build a per-channel histogram of the
   Hue, Saturation, and Value pixels. Concatenate the three into one fixed-length
   vector (default 3 x 16 = 48 numbers). This vector is a "color fingerprint":
   it says *what mix of colors* the image contains, with zero information about
   *where* those colors are. A tan-heavy NJ plate and a blue-heavy PA plate land
   in very different regions of this 48-dim space.
2. Standardize the features and fit a plain logistic regression (class-weighted
   for imbalance -- same idea as the weighted loss in the neural net).
3. Evaluate with the SAME grouped split and SAME metrics as the ResNet script,
   so the two numbers are directly comparable.

No torch, no GPU -- runs in a second or two.

Run:
    venv311/bin/python3 baseline_color_histogram.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ----------------------------------------------------------------------------
# CONFIG  — keep these in sync with train_plate_classifier.py
# Paths anchored to this script's location (in license_plates/classifier/), so
# the data dir is one level up regardless of where you launch from.
# ----------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.dirname(_SCRIPT_DIR)  # classifier/ -> license_plates/

MANIFEST_PATH = os.path.join(_DATA_DIR, "annotations.tsv")
IMAGES_ROOT = _DATA_DIR
LABEL_COL = "label"
GROUP_COL = "plate"
FILE_COL = "file"

HIST_BINS = 16       # bins per HSV channel -> feature dim = 3 * HIST_BINS
N_SPLITS = 5
SEED = 42

# Same label convention as the neural net: positive class = non-NJ (the rare one).
NJ_INDEX = 0
NON_NJ_INDEX = 1
POSITIVE = NON_NJ_INDEX
CLASS_NAMES = {NJ_INDEX: "NJ", NON_NJ_INDEX: "non-NJ"}

# Annotation labels that are NOT training data — dropped before deriving target.
EXCLUDE_LABELS = {"discard", "skip"}


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------
def color_fingerprint(path: str, bins: int) -> np.ndarray:
    """Per-channel HSV histogram, concatenated and normalized.

    HSV (not RGB) because the signal here is fundamentally about *hue* -- tan vs
    blue -- and HSV separates hue from brightness/shadow, so the fingerprint is
    more robust to lighting than raw RGB would be.
    """
    img = Image.open(path).convert("RGB").convert("HSV")
    arr = np.asarray(img)  # shape (H, W, 3), each channel 0..255
    feats = []
    for c in range(3):  # H, S, V
        hist, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 256), density=True)
        feats.append(hist)
    return np.concatenate(feats).astype(np.float32)


# ----------------------------------------------------------------------------
# Data loading  (same rules as the neural-net script)
# ----------------------------------------------------------------------------
def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_PATH, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for col in (LABEL_COL, GROUP_COL, FILE_COL):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Present: {list(df.columns)}")

    # Drop non-training labels (discard/skip) before deriving the target, so the
    # baseline and the neural net train on exactly the same rows.
    df["label_norm"] = df[LABEL_COL].str.strip().str.lower()
    n_before = len(df)
    df = df[~df["label_norm"].isin(EXCLUDE_LABELS)].reset_index(drop=True)
    if n_before - len(df):
        print(f"Excluded {n_before - len(df)} rows with non-training labels {sorted(EXCLUDE_LABELS)}.")

    is_nj = df["label_norm"].eq("nj")
    df["target"] = np.where(is_nj, NJ_INDEX, NON_NJ_INDEX)

    exists = df[FILE_COL].apply(lambda f: os.path.exists(os.path.join(IMAGES_ROOT, str(f))))
    if (~exists).sum():
        print(f"[warn] skipping {(~exists).sum()} rows with missing images.")
    df = df[exists].reset_index(drop=True)

    n_pos = int((df["target"] == NON_NJ_INDEX).sum())
    print(f"Loaded {len(df)} training rows | NJ={len(df)-n_pos}  non-NJ={n_pos}")
    return df


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    X = np.zeros((len(df), 3 * HIST_BINS), dtype=np.float32)
    for i, f in enumerate(df[FILE_COL]):
        X[i] = color_fingerprint(os.path.join(IMAGES_ROOT, str(f)), HIST_BINS)
    return X


# ----------------------------------------------------------------------------
# Metrics  (identical convention to the neural-net script, so numbers compare)
# ----------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (prob_pos >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, prob_pos) if len(np.unique(y_true)) > 1 else float("nan")
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[POSITIVE], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[NJ_INDEX, NON_NJ_INDEX])
    return {
        "pr_auc": float(pr_auc),
        "nonNJ_precision": float(p[0]),
        "nonNJ_recall": float(r[0]),
        "nonNJ_f1": float(f1[0]),
        "accuracy": float((y_pred == y_true).mean()),
        "confusion_matrix": cm,
    }


# ----------------------------------------------------------------------------
# Main: grouped CV, same as the neural net
# ----------------------------------------------------------------------------
def main():
    df = load_manifest()
    print("Extracting color fingerprints...")
    X = build_feature_matrix(df)
    y = df["target"].to_numpy()
    groups = df[GROUP_COL].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_metrics, rows = [], []
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups)):
        # Guard against the plate leak, exactly like the neural-net script.
        overlap = set(groups[tr]) & set(groups[va])
        assert not overlap, f"Plate leak! {list(overlap)[:5]}"

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        clf.fit(X[tr], y[tr])
        prob_pos = clf.predict_proba(X[va])[:, 1]  # classes_ is [0,1] -> col 1 = non-NJ

        m = compute_metrics(y[va], prob_pos)
        fold_metrics.append(m)
        cm = m["confusion_matrix"]
        print(f"fold {fold}: PR-AUC={m['pr_auc']:.3f}  non-NJ P={m['nonNJ_precision']:.3f} "
              f"R={m['nonNJ_recall']:.3f} F1={m['nonNJ_f1']:.3f}  cm[NJ]={cm[0].tolist()} cm[nonNJ]={cm[1].tolist()}")

        for i_local, i_global in enumerate(va):
            rows.append({
                "fold": fold,
                "plate": df.iloc[i_global][GROUP_COL],
                "file": df.iloc[i_global][FILE_COL],
                "true": CLASS_NAMES[int(y[i_global])],
                "prob_nonNJ": float(prob_pos[i_local]),
                "pred@0.5": CLASS_NAMES[int(prob_pos[i_local] >= 0.5)],
            })

    pd.DataFrame(rows).to_csv("baseline_val_predictions.csv", index=False)
    print("\nWrote baseline_val_predictions.csv")

    print("\n=========== BASELINE (number to beat) ===========")
    for k in ["pr_auc", "nonNJ_precision", "nonNJ_recall", "nonNJ_f1"]:
        vals = np.array([m[k] for m in fold_metrics])
        print(f"  {k:16s}: {vals.mean():.3f} +/- {vals.std():.3f}")


if __name__ == "__main__":
    main()
