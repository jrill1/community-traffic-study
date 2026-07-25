import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

FPS              = 25.0
YOLO_CLASSES     = [2, 3, 5, 7]   # car, motorcycle, bus, truck
MIN_TRACK_FRAMES = 3
MIN_TRACK_DX     = 80

EF_STATIC_BACKGROUND_ZONES = [
    (525, 275, 649, 366),  # driveway car, bottom-right corner
]

model = None


def init_model(yolo_path):
    global model
    model = YOLO(str(yolo_path))
    return model


def reload_model(yolo_path):
    global model
    import gc
    if model is not None:
        model.predictor = None
        del model
        gc.collect()
    model = YOLO(str(yolo_path))
    return model


# ── Motion detection ─────────────────────────────────────────────────────────

def detect_motion(clip_path, diff_threshold=25):
    """Returns (scores, centroids, fps).
    centroids is Nx2 float array of (cx, cy) normalised to [0,1] per frame;
    nan where no motion pixels exist.
    """
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scores, centroids = [], []
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return np.array([]), np.full((0, 2), np.nan), fps

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff  = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
        n_px  = int(np.sum(thresh) / 255)
        scores.append(n_px)
        if n_px > 0:
            ys_idx, xs_idx = np.where(thresh > 0)
            centroids.append((float(xs_idx.mean()) / fw, float(ys_idx.mean()) / fh))
        else:
            centroids.append((np.nan, np.nan))
        prev_gray = gray

    cap.release()
    return np.array(scores), np.array(centroids), fps


def _valley_split(sf, ef, scores, fps, valley_frac=0.35, min_side_sec=1.0, min_peak_ratio=0.30):
    min_side = int(min_side_sec * fps)
    seg = scores[sf:ef+1]
    if len(seg) < min_side * 2 + 1:
        return [(sf, ef)]
    best_score, best_v = 0, None
    for v in range(min_side, len(seg) - min_side):
        left_peak  = float(seg[:v].max())
        right_peak = float(seg[v+1:].max())
        valley_val = float(seg[v])
        min_peak   = min(left_peak, right_peak)
        max_peak   = max(left_peak, right_peak)
        if valley_val > valley_frac * min_peak:
            continue
        if min_peak / max_peak < min_peak_ratio:
            continue
        depth = min_peak - valley_val
        if depth > best_score:
            best_score, best_v = depth, v
    if best_v is None:
        return [(sf, ef)]
    return [(sf, sf + best_v), (sf + best_v + 1, ef)]


def _trim_window_tails(sf, ef, scores, peak_trim_frac=0.20):
    seg = scores[sf:ef+1]
    if not len(seg):
        return sf, ef
    peak_idx    = int(np.argmax(seg))
    trim_thresh = float(seg[peak_idx]) * peak_trim_frac
    new_ef_rel  = peak_idx
    for i in range(peak_idx + 1, len(seg)):
        if seg[i] < trim_thresh:
            break
        new_ef_rel = i
    new_sf_rel = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if seg[i] < trim_thresh:
            break
        new_sf_rel = i
    return sf + new_sf_rel, sf + new_ef_rel


def _merge_overlapping(windows):
    if not windows:
        return windows
    merged = [list(windows[0])]
    for s, e in windows[1:]:
        ps, pe = merged[-1]
        overlap = max(0, min(pe, e) - max(ps, s))
        if overlap > 0.5 * min(pe - ps, e - s):
            merged[-1] = [min(ps, s), max(pe, e)]
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def find_vehicle_windows(
    scores, fps,
    peak_frac=0.25, noise_percentile=25, clip_max_pct=99,
    min_gap_sec=1.5, pad_sec=0.4, min_window_sec=1.2,
    min_peak_frac=0.40, valley_frac=0.35, min_peak_ratio=0.30,
    peak_trim_frac=0.20,
):
    if not len(scores) or scores.max() == 0:
        return [], 0, 0, 0, 0
    clip_max    = float(np.percentile(scores, clip_max_pct))
    noise_floor = float(np.percentile(scores, noise_percentile))
    dyn_range   = clip_max - noise_floor
    if dyn_range <= 0:
        return [], clip_max, noise_floor, 0, float(scores.max())
    threshold = noise_floor + dyn_range * peak_frac
    active    = scores > threshold
    segments  = []
    in_seg, seg_start = False, 0
    for i, a in enumerate(active):
        if a and not in_seg:
            seg_start, in_seg = i, True
        elif not a and in_seg:
            segments.append([seg_start, i - 1])
            in_seg = False
    if in_seg:
        segments.append([seg_start, len(scores) - 1])
    min_gap_frames = int(min_gap_sec * fps)
    merged = []
    for seg in segments:
        if merged and (seg[0] - merged[-1][1]) <= min_gap_frames:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg[:])
    split = []
    for s, e in merged:
        split.extend(_valley_split(s, e, scores, fps, valley_frac, min_peak_ratio=min_peak_ratio))
    split  = [_trim_window_tails(s, e, scores, peak_trim_frac) for s, e in split]
    pad    = int(pad_sec * fps)
    total  = len(scores)
    padded = [(max(0, s - pad), min(total - 1, e + pad)) for s, e in split]
    padded = _merge_overlapping(padded)
    min_frames = int(min_window_sec * fps)
    padded = [(s, e) for s, e in padded if (e - s + 1) >= min_frames]
    padded = [(s, e) for s, e in padded
              if (scores[s:e+1].max() - noise_floor) >= dyn_range * min_peak_frac]
    return padded, clip_max, noise_floor, threshold, float(scores.max())


# ── Track stitching ──────────────────────────────────────────────────────────

def _stitch_track_segments(track_data, max_gap_sec=2.0, max_pos_jump_px=200, max_size_ratio=2.5):
    """Merge track segments from different motion windows that belong to the same vehicle."""
    if len(track_data) < 2:
        return track_data

    segs = []
    for tid, frames in track_data.items():
        if not frames:
            continue
        frames_s = sorted(frames, key=lambda f: f[0])
        xs    = [f[1] for f in frames_s]
        span  = frames_s[-1][0] - frames_s[0][0]
        vel   = (xs[-1] - xs[0]) / span if span else 0.0
        segs.append({
            'tid':      tid,
            'frames':   frames_s,
            'f0':       frames_s[0][0],
            'f1':       frames_s[-1][0],
            'x1':       xs[-1],
            'vel':      vel,
            'avg_area': float(sum(f[3] * f[4] for f in frames_s) / len(frames_s)),
        })
    segs.sort(key=lambda s: s['f0'])

    max_gap = max_gap_sec * FPS
    changed = True
    while changed:
        changed = False
        for i in range(len(segs)):
            best_j, best_err = -1, float('inf')
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                gap = b['f0'] - a['f1']
                if gap <= 0 or gap > max_gap:
                    continue
                pred_x  = a['x1'] + a['vel'] * gap
                pos_err = abs(b['frames'][0][1] - pred_x)
                if pos_err > max_pos_jump_px:
                    continue
                ratio = (max(a['avg_area'], b['avg_area'])
                         / max(min(a['avg_area'], b['avg_area']), 1.0))
                if ratio > max_size_ratio:
                    continue
                PARK_VEL = 2.0
                b_vel = (b['frames'][-1][1] - b['frames'][0][1]) / max(b['f1'] - b['f0'], 1)
                if (abs(a['vel']) < PARK_VEL) != (abs(b_vel) < PARK_VEL):
                    continue
                if pos_err < best_err:
                    best_j, best_err = j, pos_err
            if best_j >= 0:
                a, b = segs[i], segs[best_j]
                mf   = sorted(a['frames'] + b['frames'], key=lambda f: f[0])
                xs   = [f[1] for f in mf]
                span = mf[-1][0] - mf[0][0]
                segs[i] = {
                    'tid':      a['tid'],
                    'frames':   mf,
                    'f0':       mf[0][0],
                    'f1':       mf[-1][0],
                    'x1':       xs[-1],
                    'vel':      (xs[-1] - xs[0]) / span if span else 0.0,
                    'avg_area': float(sum(f[3] * f[4] for f in mf) / len(mf)),
                }
                segs.pop(best_j)
                changed = True
                break

    return {s['tid']: s['frames'] for s in segs}


def _split_track_on_id_swap(track_data, park_vel=2.0, min_split_frames=3):
    """Split tracks where YOLO ID-swapped between a moving vehicle and a stationary one."""
    result  = {}
    next_id = max(track_data.keys()) + 1 if track_data else 0
    for tid, frames in track_data.items():
        if len(frames) < min_split_frames * 2:
            result[tid] = frames
            continue
        best_i, best_score = -1, 0.0
        for i in range(min_split_frames, len(frames) - min_split_frames + 1):
            left, right = frames[:i], frames[i:]
            l_span = left[-1][0] - left[0][0]
            r_span = right[-1][0] - right[0][0]
            if l_span == 0 or r_span == 0:
                continue
            l_vel = abs((left[-1][1] - left[0][1]) / l_span)
            r_vel = abs((right[-1][1] - right[0][1]) / r_span)
            if (l_vel > park_vel) == (r_vel > park_vel):
                continue
            score = abs(l_vel - r_vel)
            if score > best_score:
                best_score, best_i = score, i
        if best_i < 0:
            result[tid] = frames
        else:
            result[tid]     = frames[:best_i]
            result[next_id] = frames[best_i:]
            next_id += 1
    return result


# ── Core YOLO tracking loop ──────────────────────────────────────────────────

def run_yolo_windows(clip_path, windows, static_masks=None, yolo_w=640):
    """
    Run YOLO tracking within the given motion windows and return stitched tracks.

    clip_path:    path to the video file
    windows:      list of (start_frame, end_frame) tuples from find_vehicle_windows
    static_masks: list of (x1,y1,x2,y2) regions in yolo-frame coords to black out
    yolo_w:       width to resize frames to before passing to YOLO

    Returns track_data dict: tid -> list of (frame_idx, x, y, w, h, cls, conf)
    in original frame coordinates, after stitching and id-swap splitting.
    """
    if static_masks is None:
        static_masks = []

    cap      = cv2.VideoCapture(str(clip_path))
    fw_orig  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_orig  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    yolo_h   = int(fh_orig * yolo_w / fw_orig)
    yolo_scale = fw_orig / yolo_w

    track_data = defaultdict(list)
    id_offset  = 0

    for ws, we in windows:
        model.predictor = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, ws)
        for fi in range(ws, we + 1):
            ret, frame = cap.read()
            if not ret:
                continue
            frame_small = cv2.resize(frame, (yolo_w, yolo_h))
            for x1m, y1m, x2m, y2m in static_masks:
                frame_small[max(0, y1m):min(yolo_h, y2m),
                            max(0, x1m):min(yolo_w, x2m)] = 0
            r = model.track(frame_small, persist=True, classes=YOLO_CLASSES,
                            verbose=False)[0]
            if r.boxes is None or r.boxes.id is None:
                continue
            for j in range(len(r.boxes)):
                tid        = int(r.boxes.id[j]) + id_offset
                x, y, w, h = r.boxes.xywh[j].tolist()
                cls        = int(r.boxes.cls[j])
                conf       = float(r.boxes.conf[j])
                track_data[tid].append((fi,
                                        x * yolo_scale, y * yolo_scale,
                                        w * yolo_scale, h * yolo_scale,
                                        cls, conf))
        if track_data:
            id_offset = max(track_data.keys()) + 1

    cap.release()
    track_data = _stitch_track_segments(dict(track_data))
    track_data = _split_track_on_id_swap(track_data)
    return track_data


# ── Motion type classification ───────────────────────────────────────────────

def assign_motion_types(events, ef_path=None, scores=None, centroids=None, fps=None,
                        far_edge_thresh=0.75, trim_frames=10):
    """
    Classify each full event as straight/turning using EF motion centroids.
    scores/centroids/fps may be passed in to avoid re-running detect_motion
    when the caller already has them (e.g. from track_clip).
    ETW -> cx at track start; WTE -> cx at track end.
    """
    if scores is None:
        scores, centroids, fps = detect_motion(ef_path)
    if not len(scores):
        for ev in events:
            ev['motion_type'] = None
        return

    windows, *_ = find_vehicle_windows(scores, fps)
    if not windows:
        for ev in events:
            ev['motion_type'] = None
        return

    for ev in events:
        if ev.get('match') != 'full':
            ev['motion_type'] = None
            continue

        ef_start = ev['_ef']['_frames'][0][0]
        ef_end   = ev['_ef']['_frames'][-1][0]

        best_win, best_overlap = None, 0
        for i, (ws, we) in enumerate(windows):
            overlap = max(0, min(ef_end, we) - max(ef_start, ws))
            if overlap > best_overlap:
                best_overlap, best_win = overlap, i

        if best_win is None:
            ev['motion_type'] = None
            continue

        ws, we   = windows[best_win]
        win_cx   = centroids[ws:we+1, 0]
        valid    = ~np.isnan(win_cx)
        if valid.sum() < 3:
            ev['motion_type'] = None
            continue

        cx    = win_cx[valid]
        trim  = trim_frames if len(cx) > 2 * trim_frames else 0
        inner = cx[trim: len(cx) - trim] if trim else cx
        if len(inner) < 2:
            ev['motion_type'] = None
            continue

        is_wte      = ev['direction'] == 'wte'
        relevant_cx = float(inner[-1]) if is_wte else float(inner[0])
        ev['motion_type'] = 'turning' if relevant_cx < far_edge_thresh else 'straight'


# ── Speed estimation ─────────────────────────────────────────────────────────

def estimate_speed(track_frames, camera, direction, speed_cal):
    """
    Estimate vehicle speed in mph from a YOLO _frames list.
    Uses per-direction k if available, falls back to camera-wide average.
    track_frames: list of (frame_idx, x, y, w, h, cls, conf) tuples.
    Returns speed in mph, or None if insufficient data.
    """
    if len(track_frames) < 2:
        return None

    dxs = []
    for i in range(1, len(track_frames)):
        dt = track_frames[i][0] - track_frames[i - 1][0]
        if dt:
            dxs.append(abs(track_frames[i][1] - track_frames[i - 1][1]) / dt)

    if not dxs:
        return None

    cam_cal = speed_cal.get(camera, {})
    k = cam_cal.get(direction) or cam_cal.get('avg')
    if not k:
        return None

    return float(np.median(dxs)) / k
