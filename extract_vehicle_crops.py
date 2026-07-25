#!/usr/bin/env python3
"""
Extract vehicle crops for high-confidence plate events.

Reads traffic_events.tsv, finds events where max(wf_plate_conf, ef_plate_conf) > CONF_THRESHOLD,
runs YOLO on both WF and EF clips, and saves the best-positioned vehicle crop from each to:
    traffic_data/make_model/YYYY_MM_DD/{event_id}_{WF|EF}.jpg

A manifest is written/updated at traffic_data/make_model/manifest.json with the associated
clip names, plate, event_id, and bounding boxes (xywh, pixel coords in the source frame).
"""

import hashlib
import json
import sys
import cv2
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

import traffic_utils as tu

PROJECT_ROOT      = Path(__file__).resolve().parent
TSV_PATH          = PROJECT_ROOT / 'traffic_events.tsv'
MAKE_MODEL_DIR    = PROJECT_ROOT / 'traffic_data' / 'make_model'
YOLO_MODEL        = PROJECT_ROOT / 'yolov8n.pt'
CONF_THRESHOLD    = 0.95
N_SAMPLE          = 20
MAX_PAIRS_PER_PLATE = 2  # cap per unique plate across all dates (for classifier training balance)

model = tu.init_model(YOLO_MODEL)


def _bbox_score(x, y, w, h, frame_w, frame_h):
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    cut = (max(0, -x1) + max(0, x2 - frame_w) +
           max(0, -y1) + max(0, y2 - frame_h))
    completeness = max(0.0, 1.0 - cut / (2 * (w + h)))
    x_central = 1.0 if 0.25 <= (x / frame_w) <= 0.75 else 0.1
    return w * h * completeness * x_central


def best_vehicle_frame(clip_path, n_sample=N_SAMPLE):
    """
    Sample frames from clip and return (frame, bbox_xyxy, yolo_conf) for the
    best-scored vehicle detection. Returns (frame, None, None) if no vehicle found.
    """
    cap   = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None, None, None

    idxs = [int(total * i / (n_sample + 1)) for i in range(1, n_sample + 1)]
    best_frame, best_bbox_xyxy, best_conf, best_score = None, None, None, 0.0
    fallback_frame = None

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if fallback_frame is None:
            fallback_frame = frame.copy()

        fh, fw = frame.shape[:2]
        results = model(frame, classes=tu.YOLO_CLASSES, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            continue

        scores = [
            _bbox_score(*results.boxes.xywh[i].tolist(), fw, fh)
            for i in range(len(results.boxes))
        ]
        bi    = int(np.argmax(scores))
        score = scores[bi]
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            xywh       = results.boxes.xywh[bi].tolist()
            x, y, w, h = xywh
            best_bbox_xyxy = [
                max(0, int(x - w / 2)), max(0, int(y - h / 2)),
                min(fw, int(x + w / 2)), min(fh, int(y + h / 2)),
            ]
            best_conf = float(results.boxes.conf[bi])

    cap.release()
    return (best_frame if best_frame is not None else fallback_frame), best_bbox_xyxy, best_conf


def crop_vehicle(frame, bbox_xyxy):
    """Crop the vehicle region from frame using [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox_xyxy
    return frame[y1:y2, x1:x2]


def resolve_clip(row, date_folder, camera):
    col  = 'ef_clip' if camera == 'EF' else 'wf_clip'
    name = row.get(col)
    if not name or pd.isna(name):
        return None
    path = PROJECT_ROOT / 'traffic_data' / date_folder / camera / 'mp4' / name
    return path if path.exists() else None


def load_manifest(path):
    if path.exists():
        return {e['event_id']: e for e in json.loads(path.read_text())}
    return {}


def save_manifest(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(entries.values(), key=lambda e: e['event_id']), indent=2))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='Only process this date folder (e.g. 2026_06_30)')
    parser.add_argument('--reprocess', action='store_true',
                        help='Re-run even if event already in manifest')
    args = parser.parse_args()

    if not TSV_PATH.exists():
        sys.exit(f"TSV not found: {TSV_PATH}")

    manifest_path = MAKE_MODEL_DIR / 'manifest.json'
    manifest      = load_manifest(manifest_path)

    # Load make_model.tsv once; flagged rows don't count against the per-plate cap.
    make_model_tsv = MAKE_MODEL_DIR / 'make_model.tsv'
    mm_df = pd.DataFrame(columns=['plate', 'date', 'timestamp', 'flagged'])
    if make_model_tsv.exists():
        mm_df = pd.read_csv(make_model_tsv, sep='\t', dtype=str,
                            usecols=['plate', 'date', 'timestamp', 'flagged', 'api_success']).fillna('')

    # (date_norm, timestamp, plate) keys for rows excluded from cap and direction counts.
    # Only successful-lookup rows that are flagged (bad crops) are excluded — failed-lookup
    # rows (api_success=0) still count so a plate can't retry a DB miss indefinitely.
    flagged_keys: set = {
        (str(r['date']).replace('-', '_'), str(r['timestamp']), r['plate'])
        for _, r in mm_df[(mm_df['flagged'] == '1') & (mm_df['api_success'] == '1')].iterrows()
    }

    def _manifest_key(e) -> tuple:
        eid = e.get('event_id', '')
        parts = eid.split('_')
        d = parts[0] if parts else ''
        return (f"{d[:4]}_{d[4:6]}_{d[6:]}", parts[1] if len(parts) > 1 else '', e.get('plate', ''))

    # Rows excluded from cap: only successful-lookup bad-crop rows (flagged=1 AND api_success=1).
    # Failed-lookup rows (api_success=0) still count regardless of flagged status.
    cap_excluded_mm = mm_df[(mm_df['flagged'] == '1') & (mm_df['api_success'] == '1')]
    cap_counted_mm  = mm_df[~mm_df.index.isin(cap_excluded_mm.index)]

    plate_pair_count = Counter(
        e['plate'] for e in manifest.values()
        if e.get('plate') and _manifest_key(e) not in flagged_keys
    )
    for plate, count in Counter(cap_counted_mm['plate'].dropna()).items():
        plate_pair_count[plate] = max(plate_pair_count.get(plate, 0), count)

    # Per-plate count of failed-lookup flagged rows — used to annotate cap-skip log lines.
    failed_lookup_flagged = Counter(
        mm_df[(mm_df['flagged'] == '1') & (mm_df['api_success'] == '0')]['plate'].dropna()
    )

    # plate_dates: ALL rows (flagged included) — same date = same clip = same result.
    plate_dates: dict = {}
    for e in manifest.values():
        if e.get('plate') and e.get('date'):
            plate_dates.setdefault(e['plate'], set()).add(str(e['date']).replace('-', '_'))
    for _, r in mm_df.iterrows():
        if r['plate'] and r['date']:
            plate_dates.setdefault(r['plate'], set()).add(str(r['date']).replace('-', '_'))
    saved_hashes  = {
        hashlib.md5((MAKE_MODEL_DIR / c).read_bytes()).hexdigest()
        for e in manifest.values()
        for c in (e.get('wf_crop'), e.get('ef_crop'))
        if c and (MAKE_MODEL_DIR / c).exists()
    }

    df = pd.read_csv(TSV_PATH, sep='\t')
    for col in ('wf_plate_conf', 'ef_plate_conf'):
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Build (date_norm, timestamp, plate) → direction index from traffic_events.tsv
    event_dir_index: dict[tuple, str] = {}
    if 'direction' in df.columns:
        for _, r in df.iterrows():
            eid = str(r.get('event_id', ''))
            parts = eid.split('_')
            if len(parts) >= 2:
                d = parts[0]
                date_norm = f"{d[:4]}_{d[4:6]}_{d[6:]}"
                ts = parts[1]
                pl = str(r.get('plate', ''))
                direction = str(r.get('direction', '')).upper()
                if pl and direction:
                    event_dir_index[(date_norm, ts, pl)] = direction

    # Build plate → Counter of travel directions (non-flagged only).
    plate_directions: defaultdict = defaultdict(Counter)
    manifest_keys: set = set()
    for e in manifest.values():
        if e.get('plate'):
            eid = e.get('event_id', '')
            parts = eid.split('_')
            if len(parts) >= 3:
                d = parts[0]
                date_norm = f"{d[:4]}_{d[4:6]}_{d[6:]}"
                ts = parts[1]
                key = (date_norm, ts, e['plate'])
                if key not in flagged_keys:
                    plate_directions[e['plate']][parts[2].upper()] += 1
                manifest_keys.add(key)
    if event_dir_index and not cap_counted_mm.empty:
        for _, r in cap_counted_mm.iterrows():
            if r['plate'] and r['date'] and r['timestamp']:
                pl = r['plate']
                date_norm = str(r['date']).replace('-', '_')
                ts = str(r['timestamp'])
                if (date_norm, ts, pl) not in manifest_keys:
                    direction = event_dir_index.get((date_norm, ts, pl), '')
                    if direction:
                        plate_directions[pl][direction] += 1

    if manifest:
        print(f"Manifest: {len(manifest)} existing entries, {len(plate_pair_count)} unique plates, "
              f"{len(saved_hashes)} unique crop hashes")

    qualifying = df[
        (df['wf_plate_conf'] > CONF_THRESHOLD) |
        (df['ef_plate_conf'] > CONF_THRESHOLD)
    ].copy()

    if args.date:
        date_filter = args.date.replace('-', '_')
        qualifying  = qualifying[
            qualifying['date'].astype(str).str.replace('-', '_') == date_filter
        ]
        print(f"Filtered to date {date_filter}: {len(qualifying)} events")

    print(f"Found {len(qualifying)} events with plate conf > {CONF_THRESHOLD} "
          f"(of {len(df)} total)")

    saved = skipped = errors = 0

    for _, row in qualifying.iterrows():
        event_id    = row['event_id']
        date_str    = str(row['date'])
        date_folder = date_str.replace('-', '_')
        plate       = row['plate'] if pd.notna(row.get('plate')) else 'unknown'
        wf_clip     = row.get('wf_clip')
        ef_clip     = row.get('ef_clip')

        if event_id in manifest and not args.reprocess:
            print(f"  [{event_id}] already in manifest — skipping")
            skipped += 1
            continue
        current_dir = str(row.get('direction', '')).upper()
        if not args.reprocess:
            count = plate_pair_count.get(plate, 0)
            if count >= MAX_PAIRS_PER_PLATE:
                dirs = plate_directions[plate]
                # Allow a 3rd crop only if existing 2 are the same direction and this
                # one is the opposite, from a day not already represented.
                qualifies_for_n3 = (
                    count == MAX_PAIRS_PER_PLATE
                    and current_dir
                    and len(dirs) == 1
                    and current_dir not in dirs
                    and date_folder not in plate_dates.get(plate, set())
                )
                if qualifies_for_n3:
                    existing_dir = next(iter(dirs))
                    print(f"  [{event_id}] plate={plate!r} at cap but qualifies for n=3 "
                          f"(existing: {sum(dirs.values())}×{existing_dir}, new: {current_dir}) — extracting")
                else:
                    ff = failed_lookup_flagged.get(plate, 0)
                    ff_note = f", {ff} flagged failed-lookup row{'s' if ff != 1 else ''}" if ff else ""
                    print(f"  [{event_id}] plate={plate!r} at cap ({count} pairs{ff_note}) — skipping")
                    skipped += 1
                    continue
            elif date_folder in plate_dates.get(plate, set()):
                print(f"  [{event_id}] plate={plate!r} already has a pair on {date_folder} — skipping")
                skipped += 1
                continue

        out_dir = MAKE_MODEL_DIR / date_folder
        out_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            'event_id': event_id,
            'plate':    plate,
            'date':     date_str,
            'wf_clip':  wf_clip if pd.notna(wf_clip) else None,
            'ef_clip':  ef_clip if pd.notna(ef_clip) else None,
            'wf_bbox':  None,
            'ef_bbox':  None,
            'wf_crop':  None,
            'ef_crop':  None,
        }

        any_saved = False
        for camera in ('WF', 'EF'):
            clip_path = resolve_clip(row, date_folder, camera)
            if clip_path is None:
                print(f"  [{event_id}] {camera} clip not found — skipping camera")
                continue

            print(f"  [{event_id}] {camera} {clip_path.name}  plate={plate!r}", end='', flush=True)
            frame, bbox, conf = best_vehicle_frame(clip_path)

            if frame is None:
                print('  → no frames read')
                continue

            stem     = '_'.join(event_id.split('_')[:2])  # YYYYMMDD_HHMMSS
            out_path = out_dir / f"{stem}_{plate}_{camera}.jpg"

            if bbox is not None:
                crop = crop_vehicle(frame, bbox)
                if crop.size > 0:
                    crop_hash = hashlib.md5(cv2.imencode('.jpg', crop)[1].tobytes()).hexdigest()
                    if crop_hash in saved_hashes and not args.reprocess:
                        print(f'  → duplicate crop (hash {crop_hash[:8]}…) — skipping')
                        continue
                    cv2.imwrite(str(out_path), crop)
                    saved_hashes.add(crop_hash)
                    rel = str(out_path.relative_to(MAKE_MODEL_DIR))
                    entry[f'{camera.lower()}_bbox'] = bbox
                    entry[f'{camera.lower()}_crop'] = rel
                    print(f'  → {out_path.name}  bbox={bbox}  yolo_conf={conf:.3f}')
                    any_saved = True
                else:
                    print('  → bbox crop empty')
            else:
                # No vehicle detected — save the full frame as fallback
                cv2.imwrite(str(out_path), frame)
                saved_hashes.add(hashlib.md5(frame.tobytes()).hexdigest())
                rel = str(out_path.relative_to(MAKE_MODEL_DIR))
                entry[f'{camera.lower()}_crop'] = rel
                print('  → no vehicle detected; saved full frame')
                any_saved = True

        if any_saved:
            manifest[event_id] = entry
            plate_pair_count[plate] += 1
            plate_dates.setdefault(plate, set()).add(date_folder)
            if current_dir:
                plate_directions[plate][current_dir] += 1
            save_manifest(manifest_path, manifest)
            saved += 1
        else:
            errors += 1

    print(f"\nDone: {saved} saved, {skipped} skipped, {errors} errors")
    print(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
