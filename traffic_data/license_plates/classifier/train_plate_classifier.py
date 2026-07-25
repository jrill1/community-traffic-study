"""
NJ vs not-NJ license-plate classifier — starter finetuning script.

What this does
--------------
Finetunes an ImageNet-pretrained ResNet-18 to a 2-class problem:
    class 0 = NJ            (the abundant majority)
    class 1 = non-NJ        (the rare "positive" we care about catching)

It is built around the three things that actually make or break this task on a
tiny, imbalanced dataset:
  1. Grouped split by `plate` so the same vehicle can't leak across train/val.
  2. Class-weighted loss so the model can't win by always guessing NJ.
  3. Imbalance-aware metrics (PR-AUC, per-class precision/recall/F1, confusion
     matrix) instead of accuracy, which is a trap here.

Run tonight (single fold, fast):
    pip install torch torchvision scikit-learn pandas pillow numpy
    python train_plate_classifier.py

For a real, noise-aware estimate later, set RUN_ALL_FOLDS = True (trains 5
models and reports mean +/- std across folds). See CONFIG below.

Outputs (written next to this script):
    best_model_fold{k}.pt      - checkpoint (weights + config + label map)
    val_predictions.csv        - per-image probs & preds; feeds your audit /
                                 threshold-picking workflow
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision import models

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ----------------------------------------------------------------------------
# Paths are anchored to THIS script's location, so it doesn't matter which
# directory you launch from. Assumes the layout:
#     license_plates/
#       annotations.tsv
#       2026_07_15/ ...            (daily crop folders; `file` column is relative to here)
#       classifier/
#         train_plate_classifier.py   <- this file
# If you move the script, update _DATA_DIR accordingly.
# ----------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.dirname(_SCRIPT_DIR)  # classifier/ -> license_plates/


# ----------------------------------------------------------------------------
# CONFIG  — the only section you should normally need to touch
# ----------------------------------------------------------------------------
@dataclass
class Config:
    # --- data ---
    manifest_path: str = os.path.join(_DATA_DIR, "annotations.tsv")
    images_root: str = _DATA_DIR          # `file` (e.g. 2026_07_15/xxx.jpg) is joined onto this
    label_col: str = "label"              # values like "nj", "pa"; anything != "nj" -> non-NJ
    group_col: str = "plate"              # split boundary — NEVER let one plate span train & val
    file_col: str = "file"                # relative image path

    # --- model / training ---
    img_size: int = 224                   # 224 keeps pretrained weights happy; try 128 later if slow
    batch_size: int = 32
    epochs: int = 20
    freeze_backbone: bool = False         # False = full finetune: features adapt (this beat 0.865), but
                                          # higher overfit risk on 80 positives — watch train_loss vs val.
                                          # True = train only the head (safe, ~logistic reg on features).
    unfreeze_last_block: bool = False     # only used when freeze_backbone=True: unfreeze layer4 for a
                                          # middle ground between head-only and full finetune
    lr: float = 1e-3                      # head-only can take a higher LR; drops to 1e-4 if backbone trains
    weight_decay: float = 1e-4
    num_workers: int = 0                  # 0 = load images in the main process. On macOS the default
                                          # FD limit is low (256) and worker processes exhaust it; for
                                          # ~1.5k tiny crops single-process is just as fast. Only raise
                                          # this once your dataset is large AND you've bumped `ulimit -n`.

    # --- cross-validation ---
    n_splits: int = 5                     # grouped, stratified folds
    run_all_folds: bool = True            # False = fold 0 only (quick); True = all 5 + mean/std
    seed: int = 42

    # --- outputs ---
    # Everything (checkpoints + prediction CSVs) is written here, tagged by run
    # name so runs never overwrite each other. Defaults to the classifier/ folder
    # next to this script. run_name="" auto-derives frozen/lastblock/unfrozen.
    output_dir: str = _SCRIPT_DIR
    run_name: str = ""

    # --- augmentation knobs (kept MILD — hue is real signal here, don't jitter it away) ---
    aug_hue: float = 0.02                 # tiny! NJ's tan bg vs PA's blue is a genuine cue
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_saturation: float = 0.1
    aug_rotation_deg: float = 5.0
    aug_perspective_p: float = 0.3
    aug_blur_p: float = 0.2


CFG = Config()


def resolve_run_name(cfg: Config) -> str:
    """A short tag for output filenames. Explicit run_name wins; otherwise it's
    derived from the training regime so a glance at the filename tells you how
    the model was trained."""
    if cfg.run_name:
        return cfg.run_name
    if not cfg.freeze_backbone:
        return "unfrozen"          # full finetune
    if cfg.unfreeze_last_block:
        return "lastblock"         # head + layer4
    return "frozen"                # head only

# ImageNet normalization stats — required because the backbone was pretrained with them.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Label convention. Positive class = non-NJ = the thing we want to *catch*.
NJ_INDEX = 0
NON_NJ_INDEX = 1
POSITIVE = NON_NJ_INDEX
CLASS_NAMES = {NJ_INDEX: "NJ", NON_NJ_INDEX: "non-NJ"}

# Annotation labels that are NOT training data — dropped before anything else.
# These are not out-of-state plates; folding them into non-NJ would poison both
# the positive class and every metric. Compared after strip().lower().
EXCLUDE_LABELS = {"discard", "skip"}


# ----------------------------------------------------------------------------
# Reproducibility & device
# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    # MPS = Apple's Metal backend. This is what makes it fast on your M4.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def harden_against_fd_limits() -> None:
    """Guard against 'OSError: [Errno 24] Too many open files' on macOS.

    Two independent mitigations: raise the process's file-descriptor soft limit
    toward its hard cap, and tell torch to share worker tensors via the
    file-system strategy instead of passing raw descriptors. Both are no-ops if
    num_workers=0, but make it safe to turn workers on later.
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else 8192
        new_soft = min(target, 8192)
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
            print(f"Raised open-file limit: {soft} -> {new_soft}")
    except Exception as e:  # resource is Unix-only; never fatal
        print(f"[warn] could not raise file-descriptor limit: {e}")
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------
class LetterboxResize:
    """Resize preserving aspect ratio, then pad to a square.

    Plates are wide. Squashing a wide plate into 224x224 distorts the characters
    and the overall shape — both of which carry state information. Letterboxing
    keeps proportions intact and pads the rest with neutral gray.
    """

    def __init__(self, size: int, fill=(114, 114, 114)):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


def build_transforms(cfg: Config):
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    # NOTE: no horizontal flip — plate text is not left-right symmetric.
    train_tf = T.Compose([
        LetterboxResize(cfg.img_size),
        T.ColorJitter(
            brightness=cfg.aug_brightness,
            contrast=cfg.aug_contrast,
            saturation=cfg.aug_saturation,
            hue=cfg.aug_hue,
        ),
        T.RandomRotation(cfg.aug_rotation_deg, fill=114),
        T.RandomPerspective(distortion_scale=0.2, p=cfg.aug_perspective_p, fill=114),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=cfg.aug_blur_p),
        T.ToTensor(),
        normalize,
    ])

    eval_tf = T.Compose([
        LetterboxResize(cfg.img_size),
        T.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class PlateDataset(Dataset):
    """Returns (image_tensor, label, row_index).

    We return the row index (not plate/file strings) so the default collate
    stays simple, and so we can map predictions back to the manifest afterward.
    """

    def __init__(self, df: pd.DataFrame, images_root: str, transform, file_col: str):
        # reset_index so positional idx == df row we can look up later
        self.df = df.reset_index(drop=True)
        self.images_root = images_root
        self.transform = transform
        self.file_col = file_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.join(self.images_root, str(row[self.file_col]))
        # Context manager guarantees the file descriptor is released even under
        # heavy iteration — PIL otherwise keeps the handle open lazily, which is
        # a classic source of "too many open files".
        with Image.open(path) as im:
            img = im.convert("RGB")
        img = self.transform(img)
        label = int(row["target"])
        return img, label, idx


# ----------------------------------------------------------------------------
# Data loading & labeling
# ----------------------------------------------------------------------------
def load_manifest(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.manifest_path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    for col in (cfg.label_col, cfg.group_col, cfg.file_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Columns present: {list(df.columns)}")

    # Normalize labels once, then drop annotation labels that aren't training
    # data (discard/skip). Do this BEFORE deriving the target so they can't
    # leak into the non-NJ class.
    df["label_norm"] = df[cfg.label_col].str.strip().str.lower()
    n_before = len(df)
    df = df[~df["label_norm"].isin(EXCLUDE_LABELS)].reset_index(drop=True)
    n_excluded = n_before - len(df)
    if n_excluded:
        print(f"Excluded {n_excluded} rows with non-training labels {sorted(EXCLUDE_LABELS)}.")

    # Binary target on what remains: NJ (0) vs any other real state/region (1).
    is_nj = df["label_norm"].eq("nj")
    df["target"] = np.where(is_nj, NJ_INDEX, NON_NJ_INDEX)

    # Drop rows whose image is missing rather than crashing mid-epoch.
    exists = df[cfg.file_col].apply(
        lambda f: os.path.exists(os.path.join(cfg.images_root, str(f)))
    )
    missing = (~exists).sum()
    if missing:
        print(f"[warn] {missing} rows point to missing image files — skipping them.")
    df = df[exists].reset_index(drop=True)

    n_pos = int((df["target"] == NON_NJ_INDEX).sum())
    n_neg = int((df["target"] == NJ_INDEX).sum())
    print(f"Loaded {len(df)} training rows | NJ={n_neg}  non-NJ={n_pos}  "
          f"(non-NJ is {100*n_pos/max(1,len(df)):.1f}% of data)")
    return df


def class_weights_from(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights ('balanced'), computed on the TRAIN split only.

    This is what stops the model from taking the lazy 'always NJ' shortcut:
    each non-NJ mistake is penalized ~ (#NJ / #non-NJ) times more.
    """
    counts = df["target"].value_counts().to_dict()
    n = len(df)
    n_classes = 2
    weights = [n / (n_classes * counts.get(c, 1)) for c in (NJ_INDEX, NON_NJ_INDEX)]
    print(f"Class weights (train): NJ={weights[0]:.3f}  non-NJ={weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(cfg: Config, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if cfg.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        if cfg.unfreeze_last_block:
            for p in model.layer4.parameters():
                p.requires_grad = True

    # Swap the 1000-class ImageNet head for our 2-class head. Newly created
    # layers always have requires_grad=True, so the head trains regardless.
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (prob_pos >= threshold).astype(int)

    # PR-AUC (average precision) on the positive class — the headline metric for
    # imbalance. Threshold-independent, and unlike ROC-AUC it doesn't get
    # flattered by the huge negative... er, majority class.
    pr_auc = average_precision_score(y_true, prob_pos) if len(np.unique(y_true)) > 1 else float("nan")

    # Per-class precision/recall/F1, focused on the non-NJ (positive) class.
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[POSITIVE], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[NJ_INDEX, NON_NJ_INDEX])
    acc = float((y_pred == y_true).mean())

    return {
        "pr_auc": float(pr_auc),
        "nonNJ_precision": float(p[0]),
        "nonNJ_recall": float(r[0]),
        "nonNJ_f1": float(f1[0]),
        "accuracy": acc,          # reported but DON'T optimize for it
        "confusion_matrix": cm,   # rows = true [NJ, non-NJ], cols = pred [NJ, non-NJ]
    }


def print_metrics(tag: str, m: dict) -> None:
    cm = m["confusion_matrix"]
    print(f"  [{tag}] PR-AUC={m['pr_auc']:.3f} | "
          f"non-NJ  P={m['nonNJ_precision']:.3f} R={m['nonNJ_recall']:.3f} F1={m['nonNJ_f1']:.3f} | "
          f"acc={m['accuracy']:.3f}")
    print(f"        confusion [true x pred], order [NJ, non-NJ]:")
    print(f"          NJ    -> {cm[0].tolist()}")
    print(f"          nonNJ -> {cm[1].tolist()}")


# ----------------------------------------------------------------------------
# One fold: train + evaluate
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels, all_idx = [], [], []
    for imgs, labels, idx in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, POSITIVE]  # P(non-NJ)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_idx.append(idx.numpy())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels),
        np.concatenate(all_idx),
    )


def run_fold(fold, train_df, val_df, cfg, device):
    # --- Safety check: no plate may appear in both splits. This is the leak
    # that makes tiny-dataset metrics look amazing and mean nothing. ---
    overlap = set(train_df[cfg.group_col]) & set(val_df[cfg.group_col])
    assert not overlap, f"Plate leak across split! Offending plates: {list(overlap)[:5]}"

    train_tf, eval_tf = build_transforms(cfg)
    train_ds = PlateDataset(train_df, cfg.images_root, train_tf, cfg.file_col)
    val_ds = PlateDataset(val_df, cfg.images_root, eval_tf, cfg.file_col)

    # persistent_workers keeps workers alive across epochs (spawn once per fold,
    # not once per epoch) — only valid when num_workers > 0.
    persist = cfg.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=False,
                              persistent_workers=persist)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers,
                            persistent_workers=persist)

    model = build_model(cfg, device)
    weights = class_weights_from(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    lr = cfg.lr if cfg.freeze_backbone else 1e-4  # full finetune wants a gentler LR
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_pr_auc, best_state, best_val = -1.0, None, None
    print(f"\n=== Fold {fold} | train={len(train_df)}  val={len(val_df)} "
          f"(val non-NJ={int((val_df['target']==NON_NJ_INDEX).sum())}) ===")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        scheduler.step()

        probs, labels_np, idx_np = evaluate(model, val_loader, device)
        m = compute_metrics(labels_np, probs)
        print(f"epoch {epoch:2d} | train_loss={running/len(train_ds):.4f}", end="")
        print_metrics("val", m)

        # Select the best epoch by PR-AUC (threshold-independent, imbalance-robust).
        if m["pr_auc"] > best_pr_auc:
            best_pr_auc = m["pr_auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val = (probs.copy(), labels_np.copy(), idx_np.copy())

    # Save best checkpoint for this fold, tagged by run name, into output_dir.
    run_name = resolve_run_name(cfg)
    ckpt_path = os.path.join(cfg.output_dir, f"best_model_{run_name}_fold{fold}.pt")
    torch.save(
        {"state_dict": best_state, "config": asdict(cfg),
         "class_names": CLASS_NAMES, "positive_index": POSITIVE},
        ckpt_path,
    )
    print(f"  -> saved {os.path.basename(ckpt_path)} (best val PR-AUC={best_pr_auc:.3f})")

    # Build per-image prediction rows for this fold's validation set.
    probs, labels_np, idx_np = best_val
    val_reset = val_df.reset_index(drop=True)
    rows = []
    for p, y, i in zip(probs, labels_np, idx_np):
        r = val_reset.iloc[int(i)]
        rows.append({
            "fold": fold,
            "plate": r[cfg.group_col],
            "file": r[cfg.file_col],
            "true": CLASS_NAMES[int(y)],
            "prob_nonNJ": float(p),
            "pred@0.5": CLASS_NAMES[int(p >= 0.5)],
        })

    # Write this fold's predictions to its own tagged file, e.g.
    # val_predictions_unfrozen_fold0.csv — so runs and folds never collide, and
    # you can diff the same fold across runs (frozen vs unfrozen see identical plates).
    csv_path = os.path.join(cfg.output_dir, f"val_predictions_{run_name}_fold{fold}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  -> saved {os.path.basename(csv_path)} ({len(rows)} rows)")

    best_m = compute_metrics(labels_np, probs)
    return best_m, rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    set_seed(CFG.seed)
    harden_against_fd_limits()
    device = pick_device()
    run_name = resolve_run_name(CFG)
    os.makedirs(CFG.output_dir, exist_ok=True)
    print(f"Device: {device} | run='{run_name}' | outputs -> {CFG.output_dir}")

    df = load_manifest(CFG)
    if int((df["target"] == NON_NJ_INDEX).sum()) < CFG.n_splits:
        print("[warn] fewer non-NJ examples than folds — stratification will be shaky. "
              "Consider RUN_ALL_FOLDS=False for now.")

    # Grouped + stratified folds: groups=plate keeps a plate in ONE split;
    # stratify=target keeps the rare class balanced across folds.
    splitter = StratifiedGroupKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    folds = list(splitter.split(df, y=df["target"], groups=df[CFG.group_col]))

    all_metrics, all_rows = [], []
    for fold, (tr_idx, va_idx) in enumerate(folds):
        train_df = df.iloc[tr_idx].copy()
        val_df = df.iloc[va_idx].copy()
        m, rows = run_fold(fold, train_df, val_df, CFG, device)
        all_metrics.append(m)
        all_rows.extend(rows)
        if not CFG.run_all_folds:
            break  # quick single-fold run

    # Each fold already wrote its own val_predictions_{run}_fold{k}.csv above.

    # Summarize. With all folds, report mean +/- std — a single 16-example val
    # fold is noisy, so the spread is the honest picture.
    print("\n================ SUMMARY ================")
    keys = ["pr_auc", "nonNJ_precision", "nonNJ_recall", "nonNJ_f1"]
    if len(all_metrics) > 1:
        for k in keys:
            vals = np.array([m[k] for m in all_metrics])
            print(f"  {k:16s}: {vals.mean():.3f} +/- {vals.std():.3f}  (n={len(vals)} folds)")
    else:
        for k in keys:
            print(f"  {k:16s}: {all_metrics[0][k]:.3f}  (single fold — set RUN_ALL_FOLDS=True for spread)")


if __name__ == "__main__":
    main()