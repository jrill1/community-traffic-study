"""
plate_scorer.py — deploy the ResNet + color baseline side by side.

Purpose
-------
Score incoming plate crops with BOTH models and log where they agree/disagree.
Disagreements (and any non-NJ flag) are your highest-value annotation candidates
AND your real-world test of whether the ResNet's strong CV numbers hold up on
genuinely new data.

Two ways to use it:

  1. As a library, inside your pipeline:
        from plate_scorer import PlateScorer
        scorer = PlateScorer()                 # loads models once
        result = scorer.score("/path/to/crop.jpg")
        # result = {color_prob, color_pred, model_prob, model_pred, agree, priority}

  2. As a batch CLI, to score a day's folder:
        venv311/bin/python3 plate_scorer.py --input ../2026_07_16 --out scored_2026_07_16.csv

Design notes
------------
* The ResNet "model" is an ENSEMBLE of your best_model_unfrozen_fold*.pt files —
  no retraining needed. Point --models at a single .pt later (e.g. a retrained
  all-data model) and it works the same (ensemble of one).
* Eval preprocessing (image size, letterbox, normalization) is read FROM the
  checkpoint's saved config, so serving can't silently drift from training.
* The color baseline is fit once on your labeled manifest and cached to
  color_baseline.pkl; delete that file to refit after you add data.
"""

from __future__ import annotations

import os
import glob
import pickle
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# --- paths anchored to this script (expected to live in classifier/) ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.dirname(_SCRIPT_DIR)  # classifier/ -> license_plates/

DEFAULT_MODELS_GLOB = os.path.join(_SCRIPT_DIR, "best_model_unfrozen_fold*.pt")
DEFAULT_MANIFEST = os.path.join(_DATA_DIR, "annotations.tsv")
DEFAULT_BASELINE_CACHE = os.path.join(_SCRIPT_DIR, "color_baseline.pkl")

# Must match training exactly.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LETTERBOX_FILL = (114, 114, 114)

# Same convention as training/baseline: positive = non-NJ.
CLASS_NAMES = {0: "NJ", 1: "non-NJ"}
EXCLUDE_LABELS = {"discard", "skip"}
HIST_BINS = 16
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# Preprocessing — identical to the training script's EVAL transform.
# ----------------------------------------------------------------------------
class LetterboxResize:
    def __init__(self, size: int, fill=LETTERBOX_FILL):
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


def build_eval_transform(img_size: int):
    return T.Compose([
        LetterboxResize(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def color_fingerprint(path: str, bins: int = HIST_BINS) -> np.ndarray:
    """Per-channel HSV histogram — identical to baseline_color_histogram.py."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB").convert("HSV"))
    feats = [np.histogram(arr[:, :, c], bins=bins, range=(0, 256), density=True)[0]
             for c in range(3)]
    return np.concatenate(feats).astype(np.float32)


def color_fingerprint_pil(img: "Image.Image", bins: int = HIST_BINS) -> np.ndarray:
    """Same as color_fingerprint but accepts an already-open PIL image."""
    arr = np.asarray(img.convert("RGB").convert("HSV"))
    feats = [np.histogram(arr[:, :, c], bins=bins, range=(0, 256), density=True)[0]
             for c in range(3)]
    return np.concatenate(feats).astype(np.float32)


# ----------------------------------------------------------------------------
# Color baseline — fit once on the labeled manifest, cache to disk.
# ----------------------------------------------------------------------------
class ColorBaseline:
    def __init__(self, pipeline, bins: int):
        self.pipeline = pipeline
        self.bins = bins

    def prob_nonNJ(self, path: str) -> float:
        x = color_fingerprint(path, self.bins).reshape(1, -1)
        return float(self.pipeline.predict_proba(x)[0, 1])  # classes_=[0,1]; col 1 = non-NJ

    def prob_nonNJ_pil(self, img: "Image.Image") -> float:
        x = color_fingerprint_pil(img, self.bins).reshape(1, -1)
        return float(self.pipeline.predict_proba(x)[0, 1])

    @classmethod
    def load_or_fit(cls, cache_path: str, manifest_path: str, images_root: str) -> "ColorBaseline":
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                obj = pickle.load(f)
            print(f"Loaded color baseline from {os.path.basename(cache_path)}")
            return cls(obj["pipeline"], obj["bins"])

        print("Fitting color baseline on labeled manifest (one-time)...")
        df = pd.read_csv(manifest_path, sep="\t", dtype=str)
        df.columns = [c.strip() for c in df.columns]
        df["label_norm"] = df["label"].str.strip().str.lower()
        df = df[~df["label_norm"].isin(EXCLUDE_LABELS)].reset_index(drop=True)
        df = df[df["file"].apply(lambda f: os.path.exists(os.path.join(images_root, str(f))))]
        df = df.reset_index(drop=True)

        y = df["label_norm"].ne("nj").astype(int).to_numpy()  # 1 = non-NJ
        X = np.zeros((len(df), 3 * HIST_BINS), dtype=np.float32)
        for i, f in enumerate(df["file"]):
            X[i] = color_fingerprint(os.path.join(images_root, str(f)))

        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ).fit(X, y)

        with open(cache_path, "wb") as f:
            pickle.dump({"pipeline": pipe, "bins": HIST_BINS}, f)
        print(f"Fit on {len(df)} rows ({int(y.sum())} non-NJ) and cached to "
              f"{os.path.basename(cache_path)}")
        return cls(pipe, HIST_BINS)


# ----------------------------------------------------------------------------
# ResNet ensemble.
# ----------------------------------------------------------------------------
class ResNetEnsemble:
    def __init__(self, ckpt_paths, device):
        if not ckpt_paths:
            raise FileNotFoundError("No model checkpoints found — check --models glob.")
        self.device = device
        self.models = []
        self.img_size = None

        for p in ckpt_paths:
            ckpt = torch.load(p, map_location="cpu")
            img_size = int(ckpt["config"]["img_size"])
            if self.img_size is None:
                self.img_size = img_size
            elif self.img_size != img_size:
                raise ValueError("Checkpoints disagree on img_size — can't ensemble mixed configs.")

            net = models.resnet18(weights=None)  # architecture only; we load OUR weights
            net.fc = nn.Linear(net.fc.in_features, 2)
            net.load_state_dict(ckpt["state_dict"])
            net.eval().to(device)
            self.models.append(net)

        self.transform = build_eval_transform(self.img_size)
        print(f"Loaded ensemble of {len(self.models)} model(s) @ img_size={self.img_size} "
              f"on {device}")

    @torch.no_grad()
    def prob_nonNJ(self, paths):
        """Mean P(non-NJ) across the ensemble, for a list of image paths."""
        batch = []
        for p in paths:
            with Image.open(p) as im:
                batch.append(self.transform(im.convert("RGB")))
        x = torch.stack(batch).to(self.device)
        probs = torch.zeros(len(paths), device=self.device)
        for net in self.models:
            probs += torch.softmax(net(x), dim=1)[:, 1]  # P(non-NJ)
        probs /= len(self.models)
        return probs.cpu().numpy()

    @torch.no_grad()
    def prob_nonNJ_imgs(self, pil_imgs):
        """Mean P(non-NJ) across the ensemble, for a list of PIL images."""
        batch = [self.transform(im.convert("RGB")) for im in pil_imgs]
        x = torch.stack(batch).to(self.device)
        probs = torch.zeros(len(pil_imgs), device=self.device)
        for net in self.models:
            probs += torch.softmax(net(x), dim=1)[:, 1]
        probs /= len(self.models)
        return probs.cpu().numpy()


# ----------------------------------------------------------------------------
# Combined scorer.
# ----------------------------------------------------------------------------
class PlateScorer:
    def __init__(self, models_glob: str = DEFAULT_MODELS_GLOB,
                 manifest_path: str = DEFAULT_MANIFEST,
                 images_root: str = _DATA_DIR,
                 baseline_cache: str = DEFAULT_BASELINE_CACHE,
                 threshold: float = 0.5,
                 near_threshold: float = 0.15,
                 device=None):
        self.threshold = threshold
        self.near = near_threshold
        self.device = device or pick_device()
        self.baseline = ColorBaseline.load_or_fit(baseline_cache, manifest_path, images_root)
        self.ensemble = ResNetEnsemble(sorted(glob.glob(models_glob)), self.device)

    def _priority(self, color_p, model_p, agree) -> str:
        """Triage label for annotation — highest value first.

        This is the active-learning ordering: disagreements and non-NJ flags are
        worth the most per label; confidently-agreed-NJ is worth the least.
        Tune the thresholds to taste.
        """
        cp = CLASS_NAMES[int(color_p >= self.threshold)]
        mp = CLASS_NAMES[int(model_p >= self.threshold)]
        if not agree:
            return "1_disagree"
        if cp == "non-NJ":  # both say non-NJ
            return "2_both_nonNJ"
        near = (abs(color_p - 0.5) < self.near) or (abs(model_p - 0.5) < self.near)
        if near:
            return "3_near_threshold"
        return "4_agree_NJ"

    def score(self, path: str) -> dict:
        return self.score_batch([path])[0]

    def score_array(self, bgr_arr: np.ndarray) -> dict:
        """Score a BGR numpy array (OpenCV format) without writing to disk."""
        pil_img = Image.fromarray(bgr_arr[:, :, ::-1])  # BGR -> RGB
        cp = self.baseline.prob_nonNJ_pil(pil_img)
        mp = float(self.ensemble.prob_nonNJ_imgs([pil_img])[0])
        color_pred = CLASS_NAMES[int(cp >= self.threshold)]
        model_pred = CLASS_NAMES[int(mp >= self.threshold)]
        agree = color_pred == model_pred
        return {
            "color_prob":  round(float(cp), 4),
            "color_pred":  color_pred,
            "model_prob":  round(float(mp), 4),
            "model_pred":  model_pred,
            "agree":       agree,
            "priority":    self._priority(cp, mp, agree),
        }

    def score_batch(self, paths) -> list:
        paths = list(paths)
        model_probs = self.ensemble.prob_nonNJ(paths)
        out = []
        for path, mp in zip(paths, model_probs):
            cp = self.baseline.prob_nonNJ(path)
            color_pred = CLASS_NAMES[int(cp >= self.threshold)]
            model_pred = CLASS_NAMES[int(mp >= self.threshold)]
            agree = color_pred == model_pred
            out.append({
                "file": path,
                "color_prob": round(float(cp), 4),
                "color_pred": color_pred,
                "model_prob": round(float(mp), 4),
                "model_pred": model_pred,
                "agree": agree,
                "priority": self._priority(cp, mp, agree),
            })
        return out


# ----------------------------------------------------------------------------
# CLI: score a folder (or manifest) and write a log sorted by priority.
# ----------------------------------------------------------------------------
def gather_inputs(input_path: str, images_root: str):
    if os.path.isdir(input_path):
        files = []
        for root, _, names in os.walk(input_path):
            for n in names:
                if n.lower().endswith(IMAGE_EXTS):
                    files.append(os.path.join(root, n))
        return sorted(files)
    # a single image file passed directly
    if input_path.lower().endswith(IMAGE_EXTS):
        return [input_path]
    # else treat as a tsv/csv manifest with a `file` column
    df = pd.read_csv(input_path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return [os.path.join(images_root, str(f)) for f in df["file"]]


def print_result(r: dict) -> None:
    flag = "" if r["agree"] else "   <-- DISAGREE"
    print()
    print(f"  file:      {r['file']}")
    print(f"  color:     P(non-NJ)={r['color_prob']:.4f}  ->  {r['color_pred']}")
    print(f"  resnet:    P(non-NJ)={r['model_prob']:.4f}  ->  {r['model_pred']}{flag}")
    print(f"  agree:     {r['agree']}")
    print(f"  priority:  {r['priority']}")


def main():
    ap = argparse.ArgumentParser(description="Score plate crops with ResNet ensemble + color baseline.")
    ap.add_argument("--input", required=True, help="a folder of crops, or a manifest tsv with a `file` column")
    ap.add_argument("--out", default=None,
                    help="CSV output path. Batch runs default to classifier/scored.csv; "
                         "single images print to terminal and skip the CSV unless --out is given.")
    ap.add_argument("--models", default=DEFAULT_MODELS_GLOB, help="glob for checkpoint(s) to ensemble")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="labeled manifest for fitting the color baseline")
    ap.add_argument("--images-root", default=_DATA_DIR)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    scorer = PlateScorer(models_glob=args.models, manifest_path=args.manifest,
                         images_root=args.images_root, threshold=args.threshold)

    files = gather_inputs(args.input, args.images_root)
    single = len(files) == 1
    if not single:
        print(f"Scoring {len(files)} crops...")

    rows = []
    for i in range(0, len(files), args.batch_size):
        rows.extend(scorer.score_batch(files[i:i + args.batch_size]))

    # Single crop: always print the prediction right here in the terminal.
    if single:
        print_result(rows[0])

    # Where (or whether) to write a CSV:
    #   --out given      -> write there
    #   batch, no --out  -> default file in classifier/
    #   single, no --out -> terminal output is enough; don't litter the disk
    out_path = args.out
    if out_path is None and not single:
        out_path = os.path.join(_SCRIPT_DIR, "scored.csv")

    df = pd.DataFrame(rows).sort_values(["priority", "model_prob"], ascending=[True, False])
    if out_path:
        df.to_csv(out_path, index=False)
        print(f"\nWrote {out_path} ({len(df)} rows).")

    if not single:
        n_disagree = int((~df["agree"]).sum())
        n_flag = int((df["priority"] != "4_agree_NJ").sum())
        print(f"  disagreements: {n_disagree}")
        print(f"  worth reviewing (non-agree-NJ): {n_flag}")
        print("  priority breakdown:")
        for k, v in df["priority"].value_counts().sort_index().items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()