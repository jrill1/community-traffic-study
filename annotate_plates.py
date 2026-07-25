#!/usr/bin/env python3
"""
License plate annotation tool.
Run: venv311/bin/python3 annotate_plates.py
Then open http://localhost:5001
"""

import argparse
import json
import os
import re
import secrets
import subprocess
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, redirect, url_for, session, send_file, render_template_string

_parser = argparse.ArgumentParser()
_parser.add_argument('--annotate_skip', action='store_true',
                     help='Work through skipped plates instead of unannotated ones')
_parser.add_argument('--anti_template', action='store_true',
                     help='Work through unannotated plates that do NOT match the A12BCD pattern')
_parser.add_argument('--disagree', action='store_true',
                     help='Work through plates where the two state classifiers disagreed')
_parser.add_argument('--review_nj', action='store_true',
                     help='Review unannotated plates where both classifiers agree: NJ')
_parser.add_argument('--review_non_nj', action='store_true',
                     help='Review unannotated plates where both classifiers agree: non-NJ')
_args, _ = _parser.parse_known_args()
ANNOTATE_SKIP     = _args.annotate_skip
ANTI_TEMPLATE     = _args.anti_template
ANNOTATE_DISAGREE = _args.disagree
REVIEW_NJ         = _args.review_nj
REVIEW_NON_NJ     = _args.review_non_nj

_TEMPLATE_RE = re.compile(r'^[A-Z]\d{2}[A-Z]{3}$')

PROJECT_ROOT = Path(__file__).resolve().parent
PLATES_DIR   = PROJECT_ROOT / 'traffic_data' / 'license_plates'
ANN_FILE     = PLATES_DIR / 'annotations.json'
TSV_FILE     = PLATES_DIR / 'annotations.tsv'

MINI      = os.environ.get('MINI_HOST', '')
MINI_DIR  = f'{MINI}:~/Documents/traffic_project/traffic_data/license_plates/'

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))


def load_annotations():
    if ANN_FILE.exists():
        return json.loads(ANN_FILE.read_text())
    return {}


def ann_to_tsv(ann):
    rows = []
    for path, meta in ann.items():
        parts = Path(path).stem.split('_')
        if len(parts) < 5:
            continue
        date, time_, direction, plate, camera = parts[0], parts[1], parts[2], parts[3], parts[4]
        rows.append((
            date, time_, direction, plate, camera, meta.get('label', ''), path,
            str(meta.get('state_pred_resnet') or ''),
            str(meta.get('state_pred_color')  or ''),
            str(meta.get('disagree', '')),
            str(meta.get('labeled_at') or ''),
        ))
    rows.sort()
    lines = ['date\ttime\tdirection\tplate\tcamera\tlabel\tfile\tstate_pred_resnet\tstate_pred_color\tdisagree\tlabeled_at']
    lines += ['\t'.join(r) for r in rows]
    return '\n'.join(lines) + '\n'


def save_annotations(ann):
    ANN_FILE.write_text(json.dumps(ann, indent=2))
    TSV_FILE.write_text(ann_to_tsv(ann))


def all_crops():
    return sorted(
        p for p in PLATES_DIR.rglob('*.jpg')
    )


def next_unannotated(crops, ann, from_idx=0):
    for i in range(from_idx, len(crops)):
        rel = str(crops[i].relative_to(PLATES_DIR))
        if rel in ann:
            continue
        if ANTI_TEMPLATE and _TEMPLATE_RE.match(crops[i].stem.split('_')[3]):
            continue
        return i
    return len(crops)


def next_skip(crops, ann, from_idx=0):
    for i in range(from_idx, len(crops)):
        rel = str(crops[i].relative_to(PLATES_DIR))
        if ann.get(rel, {}).get('label') == 'skip':
            return i
    return len(crops)


def next_disagree(crops, ann, from_idx=0):
    for i in range(from_idx, len(crops)):
        rel = str(crops[i].relative_to(PLATES_DIR))
        entry = ann.get(rel, {})
        if entry.get('disagree') == 1 and not entry.get('label'):
            return i
    return len(crops)


def advance(crops, ann, from_idx):
    if ANNOTATE_SKIP:
        return next_skip(crops, ann, from_idx)
    if ANNOTATE_DISAGREE:
        return next_disagree(crops, ann, from_idx)
    if REVIEW_NJ:
        return next_agree(crops, ann, 'nj', from_idx)
    if REVIEW_NON_NJ:
        return next_agree(crops, ann, 'non_nj', from_idx)
    return next_unannotated(crops, ann, from_idx)


def prev_unannotated(crops, ann, from_idx):
    for i in range(from_idx - 1, -1, -1):
        rel = str(crops[i].relative_to(PLATES_DIR))
        if rel not in ann:
            return i
    return from_idx  # nowhere to go, stay put


def prev_disagree(crops, ann, from_idx):
    for i in range(from_idx - 1, -1, -1):
        rel = str(crops[i].relative_to(PLATES_DIR))
        entry = ann.get(rel, {})
        if entry.get('disagree') == 1:
            return i
    return from_idx  # nowhere to go, stay put


def prev_skip(crops, ann, from_idx):
    for i in range(from_idx - 1, -1, -1):
        rel = str(crops[i].relative_to(PLATES_DIR))
        if ann.get(rel, {}).get('label') == 'skip':
            return i
    return from_idx


def next_agree(crops, ann, pred, from_idx=0):
    for i in range(from_idx, len(crops)):
        rel = str(crops[i].relative_to(PLATES_DIR))
        entry = ann.get(rel, {})
        if (entry.get('state_pred_resnet') == pred
                and entry.get('state_pred_color') == pred
                and not entry.get('label')):
            return i
    return len(crops)


def prev_agree(crops, ann, pred, from_idx):
    for i in range(from_idx - 1, -1, -1):
        rel = str(crops[i].relative_to(PLATES_DIR))
        entry = ann.get(rel, {})
        if (entry.get('state_pred_resnet') == pred
                and entry.get('state_pred_color') == pred):
            return i
    return from_idx


HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Plate Annotator</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, sans-serif;
    background: #111;
    color: #e5e5e5;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem;
    margin: 0;
    min-height: 100vh;
  }
  h2 { margin: 0 0 0.25rem; font-size: 1rem; color: #999; font-weight: 400; }
  .progress { font-size: 0.8rem; color: #666; margin-bottom: 0.75rem; }
  .current-ann {
    font-size: 0.85rem; color: #f0b429;
    background: #2a2000; border: 1px solid #5a3a00;
    padding: 0.25rem 0.75rem; border-radius: 4px;
    margin-bottom: 0.75rem;
  }
  .plate-img {
    max-width: 95vw;
    border: 2px solid #333;
    border-radius: 4px;
    image-rendering: pixelated;
    background: #000;
  }
  .buttons {
    display: flex;
    gap: 0.6rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
    justify-content: center;
  }
  button {
    padding: 0.55rem 1.4rem;
    font-size: 1rem;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    transition: filter 0.1s;
  }
  button:hover { filter: brightness(1.15); }
  .btn-back    { background: #3a3a3a; color: #ccc; }
  .btn-next    { background: #3a3a3a; color: #ccc; }
  .btn-nj      { background: #1d4ed8; color: #fff; }
  .btn-other   { background: #6d28d9; color: #fff; }
  .btn-skip    { background: #374151; color: #d1d5db; }
  .btn-discard { background: #b91c1c; color: #fff; }
  .btn-ny      { background: #047857; color: #fff; }
  .btn-pa      { background: #b45309; color: #fff; }
  .btn-open    { background: #6d28d9; color: #fff; }
  .btn-save    { background: #1d4ed8; color: #fff; }

  .sub-panel {
    display: none;
    flex-direction: column;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.75rem;
  }
  .sub-panel.visible { display: flex; }
  .sub-row { display: flex; gap: 0.6rem; }
  .custom-row {
    display: none;
    gap: 0.5rem;
    align-items: center;
    margin-top: 0.25rem;
  }
  .custom-row.visible { display: flex; }
  input[type=text] {
    padding: 0.45rem 0.75rem;
    font-size: 1rem;
    border-radius: 5px;
    border: 1px solid #444;
    background: #222;
    color: #eee;
    width: 180px;
  }
  .shortcuts {
    margin-top: 2.5rem;
    font-size: 0.75rem;
    color: #555;
    text-align: center;
    line-height: 1.8;
  }
  kbd {
    background: #222; border: 1px solid #444;
    border-radius: 3px; padding: 0 5px;
    font-family: monospace; font-size: 0.8em;
  }
  .done {
    text-align: center;
    margin-top: 3rem;
  }
  .done h1 { font-size: 2rem; margin-bottom: 0.5rem; }
  .done p  { color: #888; }

  .btn-push {
    background: #065f46; color: #fff;
    font-size: 0.85rem; padding: 0.4rem 1rem;
  }
  .push-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.72);
    align-items: center; justify-content: center;
    z-index: 100;
  }
  .push-overlay.visible { display: flex; }
  .push-modal {
    background: #1a1a1a; border: 1px solid #333;
    border-radius: 10px; padding: 1.5rem 2rem;
    text-align: center; max-width: 550px; width: 90vw;
  }
  .push-modal h3 { margin: 0 0 0.75rem; font-size: 1rem; color: #999; font-weight: 400; }
  .push-delta { font-size: 0.9rem; margin-bottom: 1.25rem; min-height: 1.5rem; text-align: left; }
  .push-delta .pos { color: #6ee7b7; }
  .push-delta .err { color: #f87171; }
  .push-row-summary   { font-size: 1rem; font-weight: 600; margin-bottom: 0.2rem; }
  .push-row-breakdown { font-size: 0.82rem; color: #888; }
  .push-divider { border: none; border-top: 1px solid #2a2a2a; margin: 0.75rem 0; }
  .push-modal-btns { display: flex; gap: 0.75rem; justify-content: center; }
</style>
</head>
<body>

<button class="btn-push" style="position:fixed;top:1rem;right:1rem;" onclick="openPushModal()">⬆ Push</button>

<div class="push-overlay" id="push-overlay">
  <div class="push-modal">
    <h3>Push to Mini</h3>
    <div class="push-delta" id="push-delta">Loading…</div>
    <div class="push-modal-btns">
      <button class="btn-back" onclick="closePushModal()">Cancel</button>
      <button class="btn-push" id="push-confirm-btn" onclick="confirmPush()">Push</button>
    </div>
  </div>
</div>

{% if done %}
  <div class="done">
    <h1>All done!</h1>
    <p>{{ annotated }} annotated &nbsp;·&nbsp; {{ discarded }} discarded &nbsp;·&nbsp; {{ skipped }} skipped &nbsp;·&nbsp; {{ total }} total</p>
    <p style="margin-top:1rem;font-size:0.85rem;color:#555">
      Annotations saved to <code>traffic_data/license_plates/annotations.json</code>
    </p>
  </div>

{% else %}
  <h2>{{ date_folder }} &nbsp;/&nbsp; {{ filename }}</h2>
  <div class="progress">
    {{ idx + 1 }} / {{ total }}
    &nbsp;·&nbsp; {{ annotated }} annotated
    &nbsp;·&nbsp; {{ discarded }} discarded
    &nbsp;·&nbsp; {{ skipped }} skipped
    {% if anti_remaining is not none %}&nbsp;·&nbsp; {{ anti_remaining }} remaining{% endif %}
  </div>

  {% if current_ann %}
    <div class="current-ann">Annotated: {{ current_ann }}</div>
  {% endif %}

  <img class="plate-img" src="/image/{{ rel_path }}" alt="plate crop">

  <form method="POST" action="/annotate" id="form">
    <input type="hidden" name="label" id="label-input" value="">

    <div class="buttons">
      <button type="button" class="btn-back"    onclick="goBack()">← Back</button>
      <button type="button" class="btn-next"    onclick="goNext()">Next →</button>
      <button type="button" class="btn-nj"      onclick="applyLabel('nj')">NJ</button>
      <button type="button" class="btn-other"   onclick="toggleOther()" id="other-btn">Other ▾</button>
      <button type="button" class="btn-skip"    onclick="goSkip()">Skip</button>
      <button type="button" class="btn-discard" onclick="applyLabel('discard')">Discard</button>
    </div>

    <div class="sub-panel" id="sub-panel">
      <div class="sub-row">
        <button type="button" class="btn-ny"   onclick="applyLabel('ny')">NY</button>
        <button type="button" class="btn-pa"   onclick="applyLabel('pa')">PA</button>
        <button type="button" class="btn-open" onclick="showCustom()">Other…</button>
      </div>
      <div class="custom-row" id="custom-row">
        <input type="text" id="custom-text" placeholder="state or label" autocomplete="off">
        <button type="button" class="btn-save" onclick="submitCustom()">Save</button>
      </div>
    </div>
  </form>

  <div class="shortcuts">
    <kbd>←</kbd> Back &nbsp; <kbd>→</kbd> Next &nbsp; <kbd>N</kbd> NJ &nbsp; <kbd>O</kbd> Other &nbsp; <kbd>S</kbd> Skip &nbsp; <kbd>D</kbd> Discard<br>
    <kbd>Y</kbd> NY &nbsp; <kbd>P</kbd> PA &nbsp; (only after Other panel opens)
  </div>
{% endif %}

<script>
function applyLabel(label) {
  document.getElementById('label-input').value = label;
  document.getElementById('form').submit();
}
function goBack() {
  window.location.href = '/annotate?action=back';
}
function goNext() {
  window.location.href = '/annotate?action=next';
}
function goSkip() {
  applyLabel('skip');
}
function toggleOther() {
  const panel = document.getElementById('sub-panel');
  panel.classList.toggle('visible');
}
function showCustom() {
  const row = document.getElementById('custom-row');
  row.classList.add('visible');
  document.getElementById('custom-text').focus();
}
function submitCustom() {
  const val = document.getElementById('custom-text').value.trim() || 'other';
  applyLabel(val);
}

const otherOpen = () => document.getElementById('sub-panel').classList.contains('visible');

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closePushModal(); return; }
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft')              goBack();
  if (e.key === 'ArrowRight')             goNext();
  if (e.key === 'n' || e.key === 'N')     applyLabel('nj');
  if (e.key === 'd' || e.key === 'D')     applyLabel('discard');
  if (e.key === 's' || e.key === 'S')     goSkip();
  if (e.key === 'o' || e.key === 'O')     toggleOther();
  if (otherOpen()) {
    if (e.key === 'y' || e.key === 'Y')   applyLabel('ny');
    if (e.key === 'p' || e.key === 'P')   applyLabel('pa');
  }
});

// Scale plate image to 10x its natural size (capped at viewport width)
const img = document.querySelector('.plate-img');
if (img) {
  const scale = () => {
    if (img.naturalWidth > 0) {
      img.style.width = Math.min(img.naturalWidth * 5, window.innerWidth * 0.95) + 'px';
      img.style.height = 'auto';
    }
  };
  img.addEventListener('load', scale);
  if (img.complete) scale();
}

// Submit custom input on Enter
document.getElementById('custom-text')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') submitCustom();
});

function openPushModal() {
  const delta = document.getElementById('push-delta');
  const btn   = document.getElementById('push-confirm-btn');
  delta.innerHTML = 'Loading…';
  btn.textContent = 'Push';
  btn.disabled = false;
  btn.style.display = '';
  document.getElementById('push-overlay').classList.add('visible');
  fetch('/push_preview')
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        delta.innerHTML = '<span class="err">Error: ' + data.error + '</span>';
        return;
      }
      const tot = data.delta_nj + data.delta_non_nj + data.delta_discarded + data.delta_skipped;
      const fmtSum = n => n > 0 ? '<span class="pos">+' + n + '</span>' : String(n);
      const dot = ' &nbsp;·&nbsp; ';
      delta.innerHTML =
        '<div class="push-row-summary">' + fmtSum(tot) + ' rows annotated</div>' +
        '<div class="push-row-breakdown">' +
          data.delta_nj + ' NJ' + dot + data.delta_non_nj + ' non-NJ' + dot +
          data.delta_discarded + ' discarded' + dot + data.delta_skipped + ' skipped' +
        '</div>' +
        '<hr class="push-divider">' +
        '<div class="push-row-summary">' + data.total + ' rows total</div>' +
        '<div class="push-row-breakdown">' +
          data.total_nj + ' NJ' + dot + data.total_non_nj + ' non-NJ' + dot +
          data.total_discarded + ' discarded' + dot + data.total_skipped + ' skipped' + dot +
          (data.total - data.total_nj - data.total_non_nj - data.total_discarded - data.total_skipped) + ' unannotated' +
        '</div>';
    })
    .catch(() => {
      delta.innerHTML = '<span class="err">Could not reach Mini</span>';
    });
}
function closePushModal() {
  document.getElementById('push-overlay').classList.remove('visible');
}
function confirmPush() {
  const btn   = document.getElementById('push-confirm-btn');
  const delta = document.getElementById('push-delta');
  btn.textContent = 'Pushing…';
  btn.disabled = true;
  fetch('/push', {method: 'POST'})
    .then(r => r.json())
    .then(data => {
      if (data.ok) {
        delta.innerHTML = '<span class="pos">Pushed successfully.</span>';
        btn.style.display = 'none';
      } else {
        delta.innerHTML = '<span class="err">Push failed: ' + data.error + '</span>';
        btn.textContent = 'Push';
        btn.disabled = false;
      }
    });
}
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return redirect(url_for('annotate'))


@app.route('/annotate', methods=['GET'])
def annotate():
    crops = all_crops()
    if not crops:
        return "No crops found in traffic_data/license_plates/", 404

    ann    = load_annotations()
    action = request.args.get('action')
    if ANNOTATE_SKIP:
        mode = 'skip'
    elif ANTI_TEMPLATE:
        mode = 'anti_template'
    elif ANNOTATE_DISAGREE:
        mode = 'disagree'
    elif REVIEW_NJ:
        mode = 'review_nj'
    elif REVIEW_NON_NJ:
        mode = 'review_non_nj'
    else:
        mode = 'normal'

    # Reset position when mode changes (e.g. restarting with --annotate_skip)
    if session.get('mode') != mode:
        session['mode'] = mode
        session.pop('idx', None)

    if action == 'back':
        cur = session.get('idx', 0)
        if ANNOTATE_DISAGREE:
            idx = prev_disagree(crops, ann, from_idx=cur)
        elif ANNOTATE_SKIP:
            idx = prev_skip(crops, ann, from_idx=cur)
        elif REVIEW_NJ:
            idx = prev_agree(crops, ann, 'nj', from_idx=cur)
        elif REVIEW_NON_NJ:
            idx = prev_agree(crops, ann, 'non_nj', from_idx=cur)
        else:
            idx = max(0, cur - 1)
    elif action == 'next':
        idx = advance(crops, ann, from_idx=session.get('idx', 0) + 1)
    elif 'idx' not in session:
        idx = advance(crops, ann, from_idx=0)
    else:
        idx = session['idx']
        # Verify cached idx is still valid for the current mode.
        if ANNOTATE_SKIP:
            rel = str(crops[idx].relative_to(PLATES_DIR)) if idx < len(crops) else None
            if rel is None or ann.get(rel, {}).get('label') != 'skip':
                idx = advance(crops, ann, from_idx=0)
        elif ANNOTATE_DISAGREE:
            rel = str(crops[idx].relative_to(PLATES_DIR)) if idx < len(crops) else None
            entry = ann.get(rel, {}) if rel else {}
            if rel is None or entry.get('disagree') != 1 or entry.get('label'):
                idx = advance(crops, ann, from_idx=0)
        elif REVIEW_NJ or REVIEW_NON_NJ:
            pred = 'nj' if REVIEW_NJ else 'non_nj'
            rel = str(crops[idx].relative_to(PLATES_DIR)) if idx < len(crops) else None
            entry = ann.get(rel, {}) if rel else {}
            if (rel is None
                    or entry.get('state_pred_resnet') != pred
                    or entry.get('state_pred_color') != pred
                    or entry.get('label')):
                idx = advance(crops, ann, from_idx=0)
        elif ANTI_TEMPLATE:
            rel = str(crops[idx].relative_to(PLATES_DIR)) if idx < len(crops) else None
            plate = crops[idx].stem.split('_')[3] if idx < len(crops) else None
            if rel is None or rel in ann or (plate and _TEMPLATE_RE.match(plate)):
                idx = advance(crops, ann, from_idx=0)

    session['idx'] = idx

    annotated = sum(1 for v in ann.values() if v['label'] and v['label'] not in ('discard', 'skip'))
    discarded = sum(1 for v in ann.values() if v['label'] == 'discard')
    skipped   = sum(1 for v in ann.values() if v['label'] == 'skip')
    anti_remaining = sum(
        1 for c in crops
        if str(c.relative_to(PLATES_DIR)) not in ann
        and not _TEMPLATE_RE.match(c.stem.split('_')[3])
    ) if ANTI_TEMPLATE else None

    if idx >= len(crops):
        return render_template_string(HTML, done=True, total=len(crops),
                                      annotated=annotated, discarded=discarded,
                                      skipped=skipped, anti_remaining=anti_remaining)

    crop     = crops[idx]
    rel_path = str(crop.relative_to(PLATES_DIR))
    current_ann = ann.get(rel_path, {}).get('label')

    return render_template_string(
        HTML,
        done=False,
        filename=crop.name,
        date_folder=crop.parent.name,
        rel_path=rel_path,
        idx=idx,
        total=len(crops),
        annotated=annotated,
        discarded=discarded,
        skipped=skipped,
        current_ann=current_ann,
        anti_remaining=anti_remaining,
    )


@app.route('/annotate', methods=['POST'])
def annotate_post():
    crops = all_crops()
    ann   = load_annotations()
    idx   = session.get('idx', 0)

    if idx < len(crops):
        label = request.form.get('label', '').strip()
        rel_path = str(crops[idx].relative_to(PLATES_DIR))
        if label:
            ann[rel_path] = {**ann.get(rel_path, {}), 'label': label,
                             'labeled_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}
            save_annotations(ann)

    session['idx'] = advance(crops, ann, from_idx=idx + 1)
    return redirect(url_for('annotate'))


@app.route('/image/<path:rel_path>')
def serve_image(rel_path):
    img_path = (PLATES_DIR / rel_path).resolve()
    if not str(img_path).startswith(str(PLATES_DIR)):
        return "Forbidden", 403
    return send_file(img_path, mimetype='image/jpeg')


@app.route('/push_preview')
def push_preview():
    ann   = load_annotations()
    total = len(all_crops())
    local_nj        = sum(1 for v in ann.values() if v['label'] == 'nj')
    local_non_nj    = sum(1 for v in ann.values() if v['label'] and v['label'] not in ('nj', 'discard', 'skip'))
    local_discarded = sum(1 for v in ann.values() if v['label'] == 'discard')
    local_skipped   = sum(1 for v in ann.values() if v['label'] == 'skip')
    try:
        result = subprocess.run(
            ['ssh', MINI,
             "awk -F'\\t' 'NR>1{l[$6]++} END{for(k in l) print k, l[k]}'"
             " ~/Documents/traffic_project/traffic_data/license_plates/annotations.tsv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return jsonify({'error': result.stderr.strip() or 'SSH failed'})
        remote = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) == 2:
                remote[parts[0]] = int(parts[1])
        remote_nj        = remote.get('nj', 0)
        remote_non_nj    = sum(v for k, v in remote.items() if k not in ('nj', 'discard', 'skip'))
        remote_discarded = remote.get('discard', 0)
        remote_skipped   = remote.get('skip', 0)
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'SSH timed out'})
    except Exception as e:
        return jsonify({'error': str(e)})
    return jsonify({
        'delta_nj':        local_nj        - remote_nj,
        'delta_non_nj':    local_non_nj    - remote_non_nj,
        'delta_discarded': local_discarded  - remote_discarded,
        'delta_skipped':   local_skipped    - remote_skipped,
        'total':           total,
        'total_nj':        local_nj,
        'total_non_nj':    local_non_nj,
        'total_discarded': local_discarded,
        'total_skipped':   local_skipped,
    })


@app.route('/push', methods=['POST'])
def push():
    try:
        for fname in ('annotations.json', 'annotations.tsv'):
            result = subprocess.run(
                ['rsync', '-az', str(PLATES_DIR / fname), MINI_DIR + fname],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                return jsonify({'ok': False, 'error': result.stderr.strip() or f'rsync failed for {fname}'})
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


if __name__ == '__main__':
    print(f"Plates dir : {PLATES_DIR}")
    print(f"Annotations: {ANN_FILE}")
    mode_str = ('skipped plates'   if ANNOTATE_SKIP    else
                'disagree plates'  if ANNOTATE_DISAGREE else
                'review NJ'        if REVIEW_NJ         else
                'review non-NJ'    if REVIEW_NON_NJ     else
                'unannotated plates')
    print(f"Mode       : {mode_str}")
    crops = all_crops()
    ann   = load_annotations()
    skipped_n = sum(1 for v in ann.values() if v['label'] == 'skip')
    print(f"Crops found: {len(crops)}  |  Annotated: {len(ann)}  |  Skipped: {skipped_n}")
    app.run(port=5001, debug=False)
