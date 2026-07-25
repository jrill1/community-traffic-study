#!/usr/bin/env python3
"""
Extract license plate crops for high-confidence OCR events.

Reads traffic_events.tsv, finds events where either camera's OCR confidence
exceeds CONF_THRESHOLD, re-runs ALPR on sampled frames from the winning clip,
and saves the tightest plate crop to:
    traffic_data/license_plates/YYYY_MM_DD/{event_id}_{plate}_{camera}.jpg
"""

import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from fast_alpr import ALPR

PROJECT_ROOT   = Path(__file__).resolve().parent
TSV_PATH       = PROJECT_ROOT / 'traffic_events.tsv'
CONF_THRESHOLD = 0.90
N_SAMPLE       = 20  # frames to sample per clip

alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)


def sample_frames(clip_path, n=N_SAMPLE):
    cap   = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    idxs = [int(total * i / (n + 1)) for i in range(1, n + 1)]
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    return frames


def best_plate_crop(clip_path):
    """
    Sample frames from clip, run ALPR on each full frame, and return
    (plate_crop, conf, ocr_text) for the highest-confidence plate reading.
    Returns (None, 0.0, None) if no plate is detected.
    """
    best_crop, best_conf, best_text = None, 0.0, None
    for frame_idx, frame in sample_frames(clip_path):
        try:
            results = alpr.predict(frame)
        except Exception as e:
            print(f"    ALPR error on frame {frame_idx}: {e}")
            continue
        for r in results:
            if r.ocr is None:
                continue
            conf = (float(np.mean(r.ocr.confidence))
                    if isinstance(r.ocr.confidence, list)
                    else float(r.ocr.confidence))
            if conf > best_conf:
                bb  = r.detection.bounding_box
                x1  = max(bb.x1, 0);  y1 = max(bb.y1, 0)
                x2  = min(bb.x2, frame.shape[1]);  y2 = min(bb.y2, frame.shape[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    best_conf = conf
                    best_crop = crop.copy()
                    best_text = r.ocr.text
    return best_crop, best_conf, best_text


def resolve_clip(row, date_folder, camera):
    col  = 'ef_clip' if camera == 'EF' else 'wf_clip'
    name = row.get(col)
    if not name or pd.isna(name):
        return None
    path = PROJECT_ROOT / 'traffic_data' / date_folder / camera / 'mp4' / name
    return path if path.exists() else None


def _write_tsv(annotations, tsv_path):
    """Write annotations.tsv from the JSON dict. Keeps TSV in sync with JSON."""
    rows = []
    for path, meta in annotations.items():
        parts = Path(path).stem.split('_')
        if len(parts) < 5:
            continue
        date, time_, direction, plate, camera = parts[0], parts[1], parts[2], parts[3], parts[4]
        rows.append((
            date, time_, direction, plate, camera,
            meta.get('label', ''), path,
            str(meta.get('state_pred_resnet') or ''),
            str(meta.get('state_pred_color')  or ''),
            str(meta.get('disagree', '')),
            str(meta.get('labeled_at') or ''),
        ))
    rows.sort()
    lines = ['date\ttime\tdirection\tplate\tcamera\tlabel\tfile\tstate_pred_resnet\tstate_pred_color\tdisagree\tlabeled_at']
    lines += ['\t'.join(r) for r in rows]
    Path(tsv_path).write_text('\n'.join(lines) + '\n')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='Only process this date folder (e.g. 2026_06_30)')
    args = parser.parse_args()

    if not TSV_PATH.exists():
        sys.exit(f"TSV not found: {TSV_PATH}")

    # Build set of plate texts already saved anywhere under license_plates/
    plates_dir = PROJECT_ROOT / 'traffic_data' / 'license_plates'
    # Filename format: {event_id}_{plate}_{camera}.jpg
    # event_id itself contains underscores, so the plate is always [-2] when split by '_'
    saved_plates = {
        p.stem.split('_')[-2]
        for p in plates_dir.rglob('*.jpg')
        if len(p.stem.split('_')) >= 3
    }
    if saved_plates:
        print(f"Skipping {len(saved_plates)} plate(s) already in license_plates/: "
              f"{', '.join(sorted(saved_plates))}")

    # Load annotations so we can append new entries and avoid double-adding.
    ANNOTATIONS_TSV  = plates_dir / 'annotations.tsv'
    ANNOTATIONS_JSON = plates_dir / 'annotations.json'

    if ANNOTATIONS_JSON.exists():
        with open(ANNOTATIONS_JSON) as f:
            annotations = json.load(f)
    else:
        annotations = {}

    # Use the JSON dict as the single source of truth for dedup.
    # Reading annotated_files from the TSV caused race conditions: if the annotator
    # had saved labels to the JSON since the last extract run, the TSV would be stale,
    # causing extract to re-add those entries (duplicate TSV rows + label overwrites).

    df = pd.read_csv(TSV_PATH, sep='\t')
    for col in ('wf_plate_conf', 'ef_plate_conf'):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Disagreements between the two state classifiers are always worth saving,
    # regardless of OCR confidence.
    if 'state_pred_resnet' in df.columns and 'state_pred_color' in df.columns:
        disagree_mask = (
            df['state_pred_resnet'].notna() &
            df['state_pred_color'].notna() &
            (df['state_pred_resnet'] != df['state_pred_color'])
        )
    else:
        disagree_mask = pd.Series(False, index=df.index)

    qualifying = df[
        (df['wf_plate_conf'] > CONF_THRESHOLD) |
        (df['ef_plate_conf'] > CONF_THRESHOLD) |
        disagree_mask
    ].copy()

    if args.date:
        date_filter = args.date.replace('-', '_')
        qualifying = qualifying[qualifying['date'].astype(str).str.replace('-', '_') == date_filter]
        print(f"Filtered to date {date_filter}: {len(qualifying)} events")

    n_disagree = int(disagree_mask.loc[qualifying.index].sum()) if not qualifying.empty else 0
    print(f"Found {len(qualifying)} events to process "
          f"({len(qualifying) - n_disagree} conf>{CONF_THRESHOLD}, {n_disagree} disagree override) "
          f"of {len(df)} total")

    saved = skipped = 0
    ann_rows_added = 0
    for _, row in qualifying.iterrows():
        event_id    = row['event_id']
        date_str    = str(row['date'])                  # "2026-06-04"
        date_folder = date_str.replace('-', '_')        # "2026_06_04"
        wf_conf     = float(row['wf_plate_conf'])
        ef_conf     = float(row['ef_plate_conf'])
        plate_text  = row['plate'] if pd.notna(row.get('plate')) else 'unknown'

        pred_resnet = str(row['state_pred_resnet']) if pd.notna(row.get('state_pred_resnet')) else ''
        pred_color  = str(row['state_pred_color'])  if pd.notna(row.get('state_pred_color'))  else ''
        is_disagree = bool(pred_resnet and pred_color and pred_resnet != pred_color)

        # Disagree cases bypass the plate-text dedup; all others skip if already saved.
        if not is_disagree and plate_text in saved_plates:
            print(f"  [{event_id}] plate={plate_text!r} already saved — skipping")
            skipped += 1
            continue

        # Pick the camera with higher confidence
        camera = 'EF' if ef_conf >= wf_conf else 'WF'
        clip_path = resolve_clip(row, date_folder, camera)

        # Fall back to the other camera if the winner's clip is missing
        if clip_path is None:
            fallback = 'WF' if camera == 'EF' else 'EF'
            clip_path = resolve_clip(row, date_folder, fallback)
            if clip_path is not None:
                camera = fallback

        if clip_path is None:
            print(f"  [{event_id}] Clip not found — skipping")
            skipped += 1
            continue

        out_dir = PROJECT_ROOT / 'traffic_data' / 'license_plates' / date_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{event_id}_{plate_text}_{camera}.jpg"

        disagree_tag = '  [DISAGREE]' if is_disagree else ''
        print(f"  [{event_id}] {camera} clip  plate={plate_text!r} "
              f"(ef={ef_conf:.2f} wf={wf_conf:.2f}){disagree_tag}")
        crop, conf, ocr_text = best_plate_crop(clip_path)

        if crop is None:
            print(f"    No plate detected in sampled frames")
            skipped += 1
            continue

        cv2.imwrite(str(out_path), crop)
        print(f"    -> {out_path.name}  re-ocr={ocr_text!r} conf={conf:.3f}")
        saved_plates.add(plate_text)
        saved += 1

        # Add to annotations (label left blank for human review).
        rel_path = f"{date_folder}/{out_path.name}"
        if rel_path not in annotations:
            annotations[rel_path] = {
                'label':             '',
                'state_pred_resnet': pred_resnet or None,
                'state_pred_color':  pred_color  or None,
                'disagree':          1 if is_disagree else 0,
            }
            ann_rows_added += 1

    if ann_rows_added:
        with open(ANNOTATIONS_JSON, 'w') as f:
            json.dump(annotations, f, indent=2)
        _write_tsv(annotations, ANNOTATIONS_TSV)
        print(f"  annotations: +{ann_rows_added} row(s) written")

    print(f"\nDone: {saved} saved, {skipped} skipped")


if __name__ == '__main__':
    main()
