"""
run_pipeline.py  —  headless version of traffic_events.ipynb

Usage:
    python run_pipeline.py [YYYY_MM_DD]

Defaults to today's date. Saves per-event debug images to debug/{date}/.
"""
import os
import gc
import sys
import time
import json
import logging
import cv2
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter

logging.getLogger("ultralytics").setLevel(logging.ERROR)

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"

from ultralytics import YOLO
from fast_alpr import ALPR
import traffic_utils as tu

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/Users/jrill/Documents/traffic_project")
YOLO_MODEL   = PROJECT_ROOT / "yolov8n.pt"
DB_PATH      = PROJECT_ROOT / "traffic_events.db"
TSV_PATH     = PROJECT_ROOT / "traffic_events.tsv"
DOWNLOAD_LOG = PROJECT_ROOT / "download.log"

# ── Date ──────────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    DATE = sys.argv[1]
else:
    DATE = datetime.now().strftime('%Y_%m_%d')

DATA_DIR  = PROJECT_ROOT / 'traffic_data' / DATE
DEBUG_DIR = PROJECT_ROOT / 'debug' / DATE
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

print(f"[pipeline] date={DATE}  data={DATA_DIR}  debug={DEBUG_DIR}")

if not DATA_DIR.exists():
    print(f"[pipeline] ERROR: data directory not found: {DATA_DIR}")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
WF_ETW_IS_LEFT          = True
EF_ETW_IS_LEFT          = True
TRANSIT_MIN_SEC         = 1.0
TRANSIT_MAX_SEC         = 10.0
USE_DB_CACHE            = True
USE_TRACKING_CACHE      = False
PLATE_CONF_THRESHOLD    = 0.95
TRACK_CACHE_DIR         = PROJECT_ROOT / 'track_cache'
RELOAD_MODEL_EVERY      = 10
TRACK_DEBUG             = True

MIN_TRACK_FRAMES = tu.MIN_TRACK_FRAMES
MIN_TRACK_DX     = tu.MIN_TRACK_DX
YOLO_CLASSES     = tu.YOLO_CLASSES
FPS              = tu.FPS

# ── Models ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT / 'traffic_data' / 'license_plates' / 'classifier'))
from plate_scorer import PlateScorer

model = tu.init_model(YOLO_MODEL)
print(f"[pipeline] YOLO loaded: {YOLO_MODEL.name}")

alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)
print("[pipeline] ALPR loaded")

scorer = PlateScorer()
print("[pipeline] PlateScorer loaded")

# ── Speed calibration ─────────────────────────────────────────────────────────
_speed_cal_path = PROJECT_ROOT / 'traffic_data/calibration/speed_cal.json'
with open(_speed_cal_path) as _f:
    SPEED_CAL = json.load(_f)

# ── Aliases from traffic_utils ────────────────────────────────────────────────
detect_motion        = tu.detect_motion
find_vehicle_windows = tu.find_vehicle_windows
assign_motion_types  = tu.assign_motion_types
_stitch_track_segments  = tu._stitch_track_segments
_split_track_on_id_swap = tu._split_track_on_id_swap


# ── Clip parsing ──────────────────────────────────────────────────────────────
def parse_clip(path):
    path  = Path(path)
    parts = path.parts
    date_dir = next(p for p in parts if len(p) == 10 and p[4] == '_' and p[7] == '_')
    y, m, d  = date_dir.split('_')
    date     = datetime(int(y), int(m), int(d))
    camera   = parts[parts.index(date_dir) + 1]
    time_part      = path.stem.split('[')[0]
    start_s, end_s = time_part.split('-')
    def t(s):
        h, mi, sec = s.split('.')
        return date.replace(hour=int(h), minute=int(mi), second=int(sec))
    return {'camera': camera, 'start': t(start_s), 'end': t(end_s), 'name': path.name}


def pair_clips(wf_dir, ef_dir):
    wf_meta = sorted(
        [{"path": p, **parse_clip(p)} for p in Path(wf_dir).glob("*.mp4")],
        key=lambda x: x["start"],
    )
    ef_meta = sorted(
        [{"path": p, **parse_clip(p)} for p in Path(ef_dir).glob("*.mp4")],
        key=lambda x: x["start"],
    )
    pairs   = []
    used_wf = set()
    for ef in ef_meta:
        candidates = [
            i for i, wf in enumerate(wf_meta)
            if i not in used_wf
            and wf["start"] <= ef["end"]
            and ef["start"] <= wf["end"]
        ]
        if not candidates:
            pairs.append({"wf": None, "ef": ef["path"], "flag": False})
        else:
            best = min(candidates,
                       key=lambda i: abs((wf_meta[i]["start"] - ef["start"]).total_seconds()))
            used_wf.add(best)
            pairs.append({"wf": wf_meta[best]["path"], "ef": ef["path"], "flag": len(candidates) > 1})
    for i, wf in enumerate(wf_meta):
        if i not in used_wf:
            pairs.append({"wf": wf["path"], "ef": None, "flag": False})
    return sorted(pairs, key=lambda p: parse_clip(p["wf"] or p["ef"])["start"])


# ── Debug image helpers ───────────────────────────────────────────────────────
def _bbox_score(x, y, w, h, frame_w, frame_h):
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    cut = (max(0, -x1) + max(0, x2 - frame_w) +
           max(0, -y1) + max(0, y2 - frame_h))
    completeness = max(0.0, 1.0 - cut / (2 * (w + h)))
    x_central = 1.0 if 0.25 <= (x / frame_w) <= 0.75 else 0.1
    return w * h * completeness * x_central


def best_vehicle_frame(clip_path, n_sample=15):
    cap   = cv2.VideoCapture(str(clip_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = [int(total * i / (n_sample + 1)) for i in range(1, n_sample + 1)]
    best_frame, best_box, best_score = None, None, 0
    fallback_frame = None
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if fallback_frame is None:
            fallback_frame = frame
        fh, fw = frame.shape[:2]
        results = model(frame, classes=YOLO_CLASSES, verbose=False)
        if results[0].boxes and len(results[0].boxes):
            scores = [
                _bbox_score(*results[0].boxes.xywh[i].tolist(), fw, fh)
                for i in range(len(results[0].boxes))
            ]
            bi    = int(np.argmax(scores))
            score = scores[bi]
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_box   = results[0].boxes.xywh[bi].tolist()
    cap.release()
    return (best_frame if best_frame is not None else fallback_frame), best_box


def save_pair(pair, idx):
    """Save side-by-side best-vehicle frames for a WF+EF clip pair to debug dir."""
    H = 360
    def render(path):
        if path is None:
            ph = np.full((H, 640, 3), 60, dtype=np.uint8)
            cv2.putText(ph, "NO CLIP", (180, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (160, 160, 160), 2)
            return ph
        frame, box = best_vehicle_frame(path)
        if frame is None:
            return np.zeros((H, 640, 3), dtype=np.uint8)
        h, w  = frame.shape[:2]
        scale = H / h
        frame = cv2.resize(frame, (int(w * scale), H))
        if box:
            x, y, bw, bh = box
            x1 = int((x - bw / 2) * scale);  y1 = int((y - bh / 2) * scale)
            x2 = int((x + bw / 2) * scale);  y2 = int((y + bh / 2) * scale)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        meta  = parse_clip(path)
        label = f"{meta['camera']}  {meta['start'].strftime('%H:%M:%S')}-{meta['end'].strftime('%H:%M:%S')}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

    wf_frame = render(pair["wf"])
    ef_frame = render(pair["ef"])
    max_h = max(wf_frame.shape[0], ef_frame.shape[0])
    def pad(f):
        if f.shape[0] < max_h:
            f = np.vstack([f, np.zeros((max_h - f.shape[0], f.shape[1], 3), dtype=np.uint8)])
        return f
    combined  = np.hstack([pad(wf_frame), pad(ef_frame)])
    flag_str  = "  !! FLAGGED" if pair["flag"] else ""
    match_str = "FULL" if pair["wf"] and pair["ef"] else ("WF-only" if pair["wf"] else "EF-only")
    header    = np.zeros((36, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"#{idx}  [{match_str}]{flag_str}",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    img = np.vstack([header, combined])
    out = DEBUG_DIR / f"pair_{idx:04d}.jpg"
    cv2.imwrite(str(out), img)


def save_bbox(clip_path, frame_data, label, out_path):
    """Save the frame with vehicle bbox drawn to out_path."""
    frame_idx, x, y, w, h = frame_data[0], frame_data[1], frame_data[2], frame_data[3], frame_data[4]
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)
    cap = cv2.VideoCapture(str(clip_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return
    pad = 20
    cx1 = max(0, x1 - pad);  cy1 = max(0, y1 - pad)
    cx2 = min(frame.shape[1], x2 + pad);  cy2 = min(frame.shape[0], y2 + pad)
    crop = frame[cy1:cy2, cx1:cx2]
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 255, 0), 3)
    cv2.putText(crop, label, (x1 - cx1, max(y1 - cy1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), crop)


# ── Tracking ──────────────────────────────────────────────────────────────────
def track_clip(clip_path, etw_is_left, dx_log=None, static_log=None, track_log=None):
    meta       = parse_clip(clip_path)
    clip_start = meta['start']
    mot_scores, ef_centroids, mot_fps = detect_motion(clip_path)
    windows, *_ = (
        tu.find_vehicle_windows(mot_scores, mot_fps or FPS)
        if len(mot_scores) else ([], None, None, None, None)
    )

    if TRACK_DEBUG:
        fps_used = mot_fps or FPS
        print(f"  [windows {meta['camera']} {Path(clip_path).name}]  {len(windows)} window(s)")
        for ws, we in windows:
            print(f"    f={ws}-{we}  ({(we-ws+1)/fps_used:.1f}s)  "
                  f"wall={clip_start + timedelta(seconds=ws/fps_used):%H:%M:%S}-"
                  f"{clip_start + timedelta(seconds=we/fps_used):%H:%M:%S}")

    static_masks = []
    if windows:
        yolo_w  = 640
        cap     = cv2.VideoCapture(str(clip_path))
        fw_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        yolo_h  = int(fh_orig * yolo_w / fw_orig)
        ws0     = windows[0][0]
        raw_masks = []
        for pfi in [ws0 - 1, ws0 - 3, ws0 - 5]:
            if pfi < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, pfi)
            ret, pframe = cap.read()
            if not ret:
                continue
            pframe_small = cv2.resize(pframe, (yolo_w, yolo_h))
            pr = model(pframe_small, classes=YOLO_CLASSES, verbose=False)[0]
            if pr.boxes:
                for j in range(len(pr.boxes)):
                    px, py, pw, ph = pr.boxes.xywh[j].tolist()
                    _x1, _y1 = int(px - pw/2), int(py - ph/2)
                    _x2, _y2 = int(px + pw/2), int(py + ph/2)
                    if meta['camera'] == 'EF' and _y1 < 100:
                        continue
                    raw_masks.append((_x1, _y1, _x2, _y2))
        deduped = []
        for m in raw_masks:
            mx1, my1, mx2, my2 = m
            duplicate = False
            for d in deduped:
                dx1, dy1, dx2, dy2 = d
                ix1, iy1 = max(mx1, dx1), max(my1, dy1)
                ix2, iy2 = min(mx2, dx2), min(my2, dy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter  = (ix2 - ix1) * (iy2 - iy1)
                    area_m = max(1, (mx2 - mx1) * (my2 - my1))
                    if inter / area_m > 0.5:
                        duplicate = True
                        break
            if not duplicate:
                deduped.append(m)
        static_masks = deduped
        if static_log is not None:
            bg_zones = tu.EF_STATIC_BACKGROUND_ZONES if meta['camera'] == 'EF' else []
            flaggable = [
                m for m in static_masks
                if not any(m[0] >= zx1 and m[1] >= zy1 and m[2] <= zx2 and m[3] <= zy2
                           for zx1, zy1, zx2, zy2 in bg_zones)
            ]
            if flaggable:
                static_log.append({'camera': meta['camera'], 'n_masks': len(flaggable)})
        if TRACK_DEBUG and static_masks:
            print(f"  [static_mask {meta['camera']}]  {len(static_masks)} region(s): {static_masks}")
        cap.release()

    if windows:
        track_data = tu.run_yolo_windows(clip_path, windows, static_masks)
    else:
        print(f"  WARNING: no motion windows in {Path(clip_path).name} — falling back to full clip")
        _cap   = cv2.VideoCapture(str(clip_path))
        _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _cap.release()
        track_data = tu.run_yolo_windows(clip_path, [(0, max(0, _total - 1))])

    if track_log is not None:
        clip_path_str = str(clip_path)
        if track_data:
            for tid in sorted(track_data.keys()):
                track_log.append((clip_path_str, tid, len(track_data[tid])))
        else:
            track_log.append((clip_path_str, None, 0))

    if TRACK_DEBUG:
        print(f"  [tracks {meta['camera']} {Path(clip_path).name}]")
        for _tid in sorted(track_data.keys()):
            _frames = track_data[_tid]
            _dx   = _frames[-1][1] - _frames[0][1] if _frames else 0
            _n    = len(_frames)
            _area = float(sum(f[3]*f[4] for f in _frames)/_n) if _n else 0
            _f0, _f1 = (_frames[0][0], _frames[-1][0]) if _frames else (0, 0)
            if _n < MIN_TRACK_FRAMES:
                _reason = f"drop n<{MIN_TRACK_FRAMES}"
            elif abs(_dx) < MIN_TRACK_DX:
                _reason = f"drop dx<{MIN_TRACK_DX}"
            else:
                _dir = "etw" if (_dx < 0) == etw_is_left else "wte"
                _reason = f"PASS ({_dir})"
            _xs     = [f[1] for f in _frames]
            _xrange = max(_xs) - min(_xs) if _xs else 0
            print(f"    tid={_tid}  dx={_dx:+.0f}px  x_range={_xrange:.0f}px  n={_n}  "
                  f"area={_area/1000:.0f}k  f={_f0}-{_f1}  -> {_reason}")

    vehicles = []
    for tid, frames in track_data.items():
        if len(frames) < MIN_TRACK_FRAMES:
            continue
        dx = frames[-1][1] - frames[0][1]
        if abs(dx) < MIN_TRACK_DX:
            if dx_log is not None:
                avg_area = float(sum(f[3] * f[4] for f in frames) / len(frames))
                dx_log.append({'clip': Path(clip_path).name, 'camera': meta['camera'],
                               'tid': tid, 'dx': dx, 'n_frames': len(frames), 'avg_area': avg_area})
            continue
        moving_left = dx < 0
        direction   = ('etw' if moving_left else 'wte') if etw_is_left else ('wte' if moving_left else 'etw')
        cls_counts  = defaultdict(int)
        for f in frames:
            cls_counts[f[5]] += 1
        cls_id = max(cls_counts, key=cls_counts.get)
        vehicles.append({
            'track_id':   tid,
            'camera':     meta['camera'],
            'direction':  direction,
            'entry_time': clip_start + timedelta(seconds=frames[0][0] / FPS),
            'exit_time':  clip_start + timedelta(seconds=frames[-1][0] / FPS),
            'dx':         dx,
            'n_frames':   len(frames),
            'class':      model.names[cls_id],
            'clip':       meta['name'],
            '_frames':    frames,
        })
    return vehicles


# ── Event matching ────────────────────────────────────────────────────────────
def match_events(wf_vehicles, ef_vehicles):
    events  = []
    used_ef = set()
    for wf in sorted(wf_vehicles, key=lambda v: v['entry_time']):
        best_match, best_dt = None, float('inf')
        for i, ef in enumerate(ef_vehicles):
            if i in used_ef or ef['direction'] != wf['direction']:
                continue
            dt = abs((ef['entry_time'] - wf['entry_time']).total_seconds())
            if TRANSIT_MIN_SEC <= dt <= TRANSIT_MAX_SEC and dt < best_dt:
                best_match, best_dt = (i, ef), dt
        if best_match:
            idx, ef = best_match
            used_ef.add(idx)
            events.append({
                'match': 'full', 'direction': wf['direction'], 'vehicle_class': wf['class'],
                'wf_entry': wf['entry_time'], 'wf_exit': wf['exit_time'],
                'ef_entry': ef['entry_time'], 'ef_exit': ef['exit_time'],
                'transit_sec': best_dt, 'wf_clip': wf['clip'], 'ef_clip': ef['clip'],
                '_wf': wf, '_ef': ef,
            })
        else:
            events.append({**wf, 'match': 'wf_only', 'transit_sec': None})
    matched_ef_tids = {e['_ef']['track_id'] for e in events if e.get('match') == 'full'}
    for ef in ef_vehicles:
        if ef['track_id'] not in matched_ef_tids:
            events.append({**ef, 'match': 'ef_only', 'transit_sec': None})
    return events


# ── Speed estimation ──────────────────────────────────────────────────────────
def estimate_speed(track_frames, camera, direction):
    return tu.estimate_speed(track_frames, camera, direction, SPEED_CAL)


# ── OCR ───────────────────────────────────────────────────────────────────────
_DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '7': 'Z', '8': 'B'}
_LETTER_TO_DIGIT = {'O': '0', 'I': '1', 'Z': '7', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8'}

def correct_nj_plate(text, conf, threshold=PLATE_CONF_THRESHOLD):
    if conf < threshold or not text or len(text) != 6:
        return text, None
    t     = text.upper()
    chars = list(t)
    last3_letters = all(c.isalpha() for c in t[3:])
    mid2_digits   = t[1].isdigit() and t[2].isdigit()
    if last3_letters or mid2_digits:
        if chars[0].isdigit():
            chars[0] = _DIGIT_TO_LETTER.get(chars[0], chars[0])
        for i in (1, 2):
            if chars[i].isalpha():
                chars[i] = _LETTER_TO_DIGIT.get(chars[i], chars[i])
        for i in (3, 4, 5):
            if chars[i].isdigit():
                chars[i] = _DIGIT_TO_LETTER.get(chars[i], chars[i])
        result = ''.join(chars)
        return result, 'nc' if result != t else None
    first3_letters = all(c.isalpha() for c in t[:3])
    last2_digits   = t[4].isdigit() and t[5].isdigit()
    if first3_letters and last2_digits:
        for i in (0, 1, 2, 3):
            if chars[i].isdigit():
                chars[i] = _DIGIT_TO_LETTER.get(chars[i], chars[i])
        for i in (4, 5):
            if chars[i].isalpha():
                chars[i] = _LETTER_TO_DIGIT.get(chars[i], chars[i])
        result = ''.join(chars)
        return result, 'com' if result != t else None
    return t, None


def read_plate(clip_path, track_frames, camera='WF', motion_type=None, n_frames=3):
    cap = cv2.VideoCapture(str(clip_path))
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if camera == 'EF' and motion_type == 'straight' and fw:
        right = sorted([f for f in track_frames if f[1] > fw * 0.75],
                       key=lambda f: f[3] * f[4], reverse=True)
        left  = sorted([f for f in track_frames if f[1] <= fw * 0.75],
                       key=lambda f: f[3] * f[4], reverse=True)
        candidates = (right + left)[:n_frames]
    elif camera == 'WF' and fw:
        left  = sorted([f for f in track_frames if f[1] < fw * 0.5],
                       key=lambda f: f[3] * f[4], reverse=True)
        right = sorted([f for f in track_frames if f[1] >= fw * 0.5],
                       key=lambda f: f[3] * f[4], reverse=True)
        candidates = (left + right)[:n_frames]
    else:
        candidates = sorted(track_frames, key=lambda f: f[3] * f[4], reverse=True)[:n_frames]
    frame_lookup = {f[0]: f for f in track_frames}
    readings = []
    for frame_data in candidates:
        frame_idx = frame_data[0]
        x_c, y_c, bw, bh = frame_data[1], frame_data[2], frame_data[3], frame_data[4]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        pad = int(max(bw, bh) * 0.02)
        x1  = max(0, int(x_c - bw / 2) - pad);  y1 = max(0, int(y_c - bh / 2) - pad)
        x2  = min(frame.shape[1], int(x_c + bw / 2) + pad);  y2 = min(frame.shape[0], int(y_c + bh / 2) + pad)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        try:
            results = alpr.predict(roi)
        except Exception as e:
            print(f"  [{camera}] ALPR error on frame {frame_idx}: {e}")
            continue
        if results and results[0].ocr:
            ocr  = results[0].ocr
            conf = float(np.mean(ocr.confidence)) if isinstance(ocr.confidence, list) else float(ocr.confidence)
            plate_crop = None
            try:
                bb = results[0].detection.bounding_box
                px1 = max(int(bb.x1), 0);  py1 = max(int(bb.y1), 0)
                px2 = min(int(bb.x2), roi.shape[1]);  py2 = min(int(bb.y2), roi.shape[0])
                if px2 > px1 and py2 > py1:
                    plate_crop = roi[py1:py2, px1:px2].copy()
            except Exception:
                pass
            readings.append((ocr.text, conf, frame_idx, plate_crop))
            if conf >= PLATE_CONF_THRESHOLD:
                break
    cap.release()
    if not readings:
        return None, 0.0, None, None, None, None
    best_text, best_conf, best_frame_idx, best_plate_crop = max(readings, key=lambda r: r[1])
    corrected, rule = correct_nj_plate(best_text, best_conf)
    return corrected, best_conf, frame_lookup.get(best_frame_idx), rule, best_text, best_plate_crop


# ── Database ──────────────────────────────────────────────────────────────────
CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS traffic_events (
    event_id         TEXT PRIMARY KEY,
    direction        TEXT,
    date             TEXT,
    event_start      TEXT,
    event_end        TEXT,
    wf_clip          TEXT,
    ef_clip          TEXT,
    wf_entry_time    TEXT,
    wf_exit_time     TEXT,
    ef_entry_time    TEXT,
    ef_exit_time     TEXT,
    transit_time_sec REAL,
    plate            TEXT,
    wf_plate         TEXT,
    ef_plate         TEXT,
    wf_plate_conf    REAL,
    ef_plate_conf    REAL,
    vehicle_class    TEXT,
    match_confidence REAL,
    wf_speed_mph     REAL,
    ef_speed_mph     REAL,
    speed_mph        REAL,
    motion_type          TEXT,
    contains_parked      BOOLEAN,
    state_pred_resnet  TEXT,
    state_pred_color   TEXT
)
"""

def event_to_row(ev):
    if ev['match'] != 'full':
        return None
    ts       = ev['wf_entry']
    event_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{ev['direction']}"
    return (
        event_id, ev['direction'], ts.strftime('%Y-%m-%d'),
        ev['wf_entry'].isoformat(), ev['ef_exit'].isoformat(),
        ev['wf_clip'], ev['ef_clip'],
        ev['wf_entry'].isoformat(), ev['wf_exit'].isoformat(),
        ev['ef_entry'].isoformat(), ev['ef_exit'].isoformat(),
        ev['transit_sec'], ev.get('plate'), ev.get('wf_plate'), ev.get('ef_plate'),
        ev.get('wf_plate_conf'), ev.get('ef_plate_conf'),
        ev['vehicle_class'], None,
        ev.get('wf_speed_mph'), ev.get('ef_speed_mph'), ev.get('speed_mph'),
        ev.get('motion_type'), ev.get('contains_parked', 0),
        ev.get('state_pred_resnet'), ev.get('state_pred_color'),
    )


# ── Nighttime track summary ───────────────────────────────────────────────────
def _is_nighttime():
    h = datetime.now().hour
    return h >= 21 or h < 4


def _get_nighttime_cutoff():
    """Return the previous run's timestamp from download.log as the mtime cutoff."""
    if not DOWNLOAD_LOG.exists():
        return datetime.now() - timedelta(hours=2)
    file_lines = DOWNLOAD_LOG.read_text().splitlines(keepends=True)
    border_indices = [
        i for i, l in enumerate(file_lines)
        if l.strip() and all(c == '═' for c in l.strip())
    ]
    if len(border_indices) >= 4:
        prev_idx = border_indices[-4]
        if prev_idx + 1 < len(file_lines):
            try:
                return datetime.strptime(file_lines[prev_idx + 1].strip(), '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
    return datetime.now() - timedelta(hours=2)


def _build_track_lines(log, cutoff_dt=None):
    lines = []
    for clip_path_str, tid, n_frames in log:
        if cutoff_dt is not None:
            try:
                mtime = datetime.fromtimestamp(Path(clip_path_str).stat().st_mtime)
                if mtime < cutoff_dt:
                    continue
            except OSError:
                continue
        meta = parse_clip(clip_path_str)
        label = f"{meta['start']:%H:%M}-{meta['end']:%H:%M}"
        if tid is None:
            lines.append(f'    {label} | No Tracks :(\n')
        else:
            n_word = 'frame' if n_frames == 1 else 'frames'
            lines.append(f'    {label} | Track {tid} | {n_frames} {n_word}\n')
    return lines


def _inject_nighttime_tracks(wf_log, ef_log):
    if not DOWNLOAD_LOG.exists():
        return

    content = DOWNLOAD_LOG.read_text()
    file_lines = content.splitlines(keepends=True)

    border_indices = [
        i for i, l in enumerate(file_lines)
        if l.strip() and all(c == '═' for c in l.strip())
    ]

    if len(border_indices) < 2:
        return

    open_idx  = border_indices[-2]
    close_idx = border_indices[-1]

    cutoff_dt = _get_nighttime_cutoff()
    wf_lines = _build_track_lines(wf_log, cutoff_dt)
    ef_lines = _build_track_lines(ef_log, cutoff_dt)

    if not wf_lines and not ef_lines:
        return

    new_lines = list(file_lines)
    offset = 0
    for i in range(open_idx + 1, close_idx):
        s = file_lines[i].lstrip()
        if s.startswith('WF '):
            pos = i + 1 + offset
            for j, l in enumerate(wf_lines):
                new_lines.insert(pos + j, l)
            offset += len(wf_lines)
        elif s.startswith('EF '):
            pos = i + 1 + offset
            for j, l in enumerate(ef_lines):
                new_lines.insert(pos + j, l)
            offset += len(ef_lines)

    DOWNLOAD_LOG.write_text(''.join(new_lines))


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    t_start = time.perf_counter()
    #is_night = _is_nighttime()
    is_night = True
    wf_track_log, ef_track_log = [], []

    pairs = pair_clips(DATA_DIR / 'WF' / 'mp4', DATA_DIR / 'EF' / 'mp4')
    full    = sum(1 for p in pairs if p['wf'] and p['ef'])
    wf_only = sum(1 for p in pairs if p['wf'] and not p['ef'])
    ef_only = sum(1 for p in pairs if p['ef'] and not p['wf'])
    flagged = sum(1 for p in pairs if p['flag'])
    print(f"[pipeline] {len(pairs)} pairs — {full} full | {wf_only} WF-only | {ef_only} EF-only | {flagged} flagged")

    _cal_clips  = {p.name for p in (PROJECT_ROOT / 'traffic_data' / 'calibration').rglob('*.mp4')}
    data_pairs  = [p for p in pairs if p['wf'] and p['ef']]
    _before     = len(data_pairs)
    data_pairs  = [p for p in data_pairs if Path(p['wf']).name not in _cal_clips
                   and Path(p['ef']).name not in _cal_clips]
    if _before - len(data_pairs):
        print(f"[pipeline] Excluded {_before - len(data_pairs)} calibration pair(s)")

    if DB_PATH.exists():
        DB_PATH.chmod(0o644)

    con = sqlite3.connect(DB_PATH)
    con.execute(CREATE_TABLE)
    for col in ('wf_speed_mph REAL', 'ef_speed_mph REAL', 'speed_mph REAL', 'motion_type TEXT',
                'wf_plate_conf REAL', 'ef_plate_conf REAL', 'contains_parked BOOLEAN',
                'state_pred_resnet TEXT', 'state_pred_color TEXT'):
        try:
            con.execute(f"ALTER TABLE traffic_events ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass

    try:
        import psutil as _psutil
        def _mem_gb(): return _psutil.Process().memory_info().rss / 1e9
    except ImportError:
        def _mem_gb(): return float('nan')

    full_event_count = 0
    agg_motion = agg_track = agg_alpr = agg_scorer = agg_event_processing = 0.0
    global model

    for pair_num, pair in enumerate(data_pairs, 1):
        if RELOAD_MODEL_EVERY and pair_num > 1 and (pair_num - 1) % RELOAD_MODEL_EVERY == 0:
            model = tu.reload_model(YOLO_MODEL)
            tu.model = model
            print(f"  [model reloaded at pair {pair_num}]")

        wf_path, ef_path = pair["wf"], pair["ef"]
        wf_name = Path(wf_path).name
        ef_name = Path(ef_path).name

        if USE_DB_CACHE:
            cached = con.execute(
                "SELECT COUNT(*) FROM traffic_events WHERE wf_clip = ? AND ef_clip = ?",
                (wf_name, ef_name),
            ).fetchone()[0]
            if cached:
                print(f"Pair {pair_num}: cached ({cached} event(s)) — skipping")
                continue

        t_pair = time.perf_counter()

        t0 = time.perf_counter()
        ef_scores, ef_centroids, ef_fps = detect_motion(ef_path)
        t_motion_ef = time.perf_counter() - t0

        dx_log, wf_static_log, ef_static_log = [], [], []

        t0 = time.perf_counter()
        wf_vehicles = track_clip(wf_path, WF_ETW_IS_LEFT, dx_log=dx_log, static_log=wf_static_log,
                                 track_log=wf_track_log if is_night else None)
        t_track_wf = time.perf_counter() - t0

        t0 = time.perf_counter()
        ef_vehicles = track_clip(ef_path, EF_ETW_IS_LEFT, dx_log=dx_log, static_log=ef_static_log,
                                 track_log=ef_track_log if is_night else None)
        t_track_ef = time.perf_counter() - t0

        for e in dx_log:
            print(f"  [dx_filter {e['camera']}] tid={e['tid']}  |dx|={abs(e['dx']):.0f}px  "
                  f"n_frames={e['n_frames']}  avg_bbox={e['avg_area']:.0f}px²")

        contains_parked = int(
            bool(wf_static_log or ef_static_log) or
            any(abs(e['dx']) < 5 and e['n_frames'] >= 10 and e['avg_area'] > 50_000 for e in dx_log)
        )

        events = match_events(wf_vehicles, ef_vehicles)
        del wf_vehicles, ef_vehicles

        _dirs  = {ev['direction'] for ev in events}
        _n     = len(events)
        if _n == 0:             _dir_tag = '[  ]'
        elif _n == 1:           _dir_tag = '[→]' if 'wte' in _dirs else '[←]'
        elif _dirs == {'wte'}: _dir_tag = '[→→]'
        elif _dirs == {'etw'}: _dir_tag = '[←←]'
        else:                   _dir_tag = '[⇄]'
        print(f"\n{_dir_tag} Pair {pair_num}: {wf_name}  +  {ef_name}")

        # Save pair image for offline review
        save_pair(pair, pair_num)

        for ev in events:
            ev['contains_parked'] = contains_parked

        for ev in events:
            if ev.get('match') == 'full':
                wf_spd = estimate_speed(ev['_wf']['_frames'], 'WF', ev['direction'])
                ef_spd = estimate_speed(ev['_ef']['_frames'], 'EF', ev['direction'])
                spds   = [s for s in [wf_spd, ef_spd] if s is not None]
                ev['wf_speed_mph'] = round(wf_spd, 1) if wf_spd is not None else None
                ev['ef_speed_mph'] = round(ef_spd, 1) if ef_spd is not None else None
                ev['speed_mph']    = round(float(np.mean(spds)), 1) if spds else None
            else:
                ev['wf_speed_mph'] = ev['ef_speed_mph'] = ev['speed_mph'] = None

        assign_motion_types(events, ef_path, scores=ef_scores,
                            centroids=ef_centroids, fps=ef_fps)
        del ef_scores, ef_centroids

        for ev in events:
            if ev.get('match') == 'full' and ev.get('motion_type') == 'turning':
                ev['speed_mph'] = ev['wf_speed_mph']

        t_alpr_total = t_scorer_total = 0.0
        for ev in events:
            if ev["match"] != "full":
                ev["wf_plate"] = ev["ef_plate"] = ev["plate"] = None
                ev["wf_plate_conf"] = ev["ef_plate_conf"] = 0.0
                continue

            t0 = time.perf_counter()
            ef_plate, ef_conf, ef_best, ef_rule, ef_raw, ef_crop = read_plate(
                ef_path, ev["_ef"]["_frames"], camera='EF', motion_type=ev.get('motion_type'))
            t_alpr_total += time.perf_counter() - t0

            if ef_conf >= PLATE_CONF_THRESHOLD:
                wf_plate, wf_conf, wf_best, wf_rule, wf_raw, wf_crop = None, 0.0, None, None, None, None
            else:
                t0 = time.perf_counter()
                wf_plate, wf_conf, wf_best, wf_rule, wf_raw, wf_crop = read_plate(
                    wf_path, ev["_wf"]["_frames"], camera='WF')
                t_alpr_total += time.perf_counter() - t0

            if ef_conf >= wf_conf:
                plate, best_frame, best_clip, best_conf, plate_rule, best_crop = ef_plate, ef_best, ef_path, ef_conf, ef_rule, ef_crop
            else:
                plate, best_frame, best_clip, best_conf, plate_rule, best_crop = wf_plate, wf_best, wf_path, wf_conf, wf_rule, wf_crop

            ev["wf_plate"] = wf_plate;  ev["ef_plate"] = ef_plate
            ev["wf_plate_conf"] = wf_conf;  ev["ef_plate_conf"] = ef_conf
            ev["plate"] = plate

            ev["state_pred_resnet"] = ev["state_pred_color"] = None
            if best_crop is not None and best_crop.size > 0:
                try:
                    t0 = time.perf_counter()
                    sr = scorer.score_array(best_crop)
                    t_scorer_total += time.perf_counter() - t0
                    ev["state_pred_resnet"] = sr["model_pred"].lower().replace("-", "_")
                    ev["state_pred_color"]  = sr["color_pred"].lower().replace("-", "_")
                except Exception as _e:
                    print(f"  [scorer] error: {_e}")

            parked_tag = ' [parked]' if contains_parked else ''
            rule_tag   = f'  [corrected: {plate_rule}]' if plate_rule else ''
            t_now = time.perf_counter()
            print(f"  [{ev['direction']}] [{ev.get('motion_type') or '?'}]{parked_tag}  "
                  f"WF: {wf_raw!r} ({wf_conf:.2f})  "
                  f"EF: {ef_raw!r} ({ef_conf:.2f})  "
                  f"-> {ev['plate']!r}{rule_tag}  |  {ev['speed_mph']} mph  |  {t_now - t_start:.1f}s elapsed")

            # Save bbox image for offline plate review
            if best_frame is not None:
                ts       = ev['wf_entry']
                event_id = f"{ts.strftime('%H%M%S')}_{ev['direction']}"
                plate_str = plate or 'no_plate'
                img_path  = DEBUG_DIR / f"pair_{pair_num:04d}_{event_id}_{plate_str}.jpg"
                save_bbox(best_clip, best_frame, f"{plate} ({best_conf:.2f})", img_path)

        t_pair_total       = time.perf_counter() - t_pair
        t_event_processing = t_pair_total - t_motion_ef - t_track_wf - t_track_ef - t_alpr_total - t_scorer_total
        agg_motion            += t_motion_ef
        agg_track             += t_track_wf + t_track_ef
        agg_alpr              += t_alpr_total
        agg_scorer            += t_scorer_total
        agg_event_processing  += t_event_processing
        print(f"  [timing]  motion_ef={t_motion_ef:.1f}s  track_wf={t_track_wf:.1f}s  "
              f"track_ef={t_track_ef:.1f}s  alpr={t_alpr_total:.1f}s  scorer={t_scorer_total:.1f}s  "
              f"event_processing={t_event_processing:.1f}s  total={t_pair_total:.1f}s  mem={_mem_gb():.1f}g")

        rows = [row for ev in events if (row := event_to_row(ev)) is not None]
        if rows:
            con.executemany(
                """INSERT OR REPLACE INTO traffic_events
                   (event_id, direction, date, event_start, event_end,
                    wf_clip, ef_clip, wf_entry_time, wf_exit_time, ef_entry_time, ef_exit_time,
                    transit_time_sec, plate, wf_plate, ef_plate,
                    wf_plate_conf, ef_plate_conf,
                    vehicle_class, match_confidence,
                    wf_speed_mph, ef_speed_mph, speed_mph,
                    motion_type, contains_parked,
                    state_pred_resnet, state_pred_color)
                   VALUES (?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?, ?,?, ?,?, ?,?,?, ?,?,?,?)""",
                rows,
            )
            con.commit()
            pd.read_sql("SELECT * FROM traffic_events", sqlite3.connect(DB_PATH)).to_csv(
                TSV_PATH, sep="\t", index=False
            )

        full_event_count += sum(1 for e in events if e.get("match") == "full")
        del events
        gc.collect()

    con.close()
    if is_night:
        # Also track any newly-downloaded clips that weren't part of a full pair
        # (EF-only when WF has no clips, or when a newer clip lost the WF slot to an older one)
        _night_cutoff = _get_nighttime_cutoff()
        for p in pairs:
            try:
                if p.get('ef') and not p.get('wf'):
                    ef_p = p['ef']
                    if Path(ef_p).name not in _cal_clips and \
                       datetime.fromtimestamp(Path(ef_p).stat().st_mtime) > _night_cutoff:
                        track_clip(ef_p, EF_ETW_IS_LEFT, track_log=ef_track_log)
                elif p.get('wf') and not p.get('ef'):
                    wf_p = p['wf']
                    if Path(wf_p).name not in _cal_clips and \
                       datetime.fromtimestamp(Path(wf_p).stat().st_mtime) > _night_cutoff:
                        track_clip(wf_p, WF_ETW_IS_LEFT, track_log=wf_track_log)
            except OSError:
                pass
        _inject_nighttime_tracks(wf_track_log, ef_track_log)
    t_total = time.perf_counter() - t_start
    print(f"\n{'=' * 30}")
    print(f"[pipeline] {full_event_count} full events  |  debug images -> {DEBUG_DIR}")
    print(f"[timing:total]  motion={agg_motion:.1f}s  track={agg_track:.1f}s  alpr={agg_alpr:.1f}s  "
          f"scorer={agg_scorer:.1f}s  event_processing={agg_event_processing:.1f}s")
    print(f"  TOTAL={t_total:.0f}s")
    print(f"[pipeline:summary] {full_event_count} events | {t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
