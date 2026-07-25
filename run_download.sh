#!/bin/bash
source /Users/jrill/.zprofile
CAMERA_PASSWORD=$(security find-generic-password -a camera -s traffic_project -w 2>/dev/null)
PUSHOVER_USER_KEY=$(security find-generic-password -a pushover_user -s traffic_project -w 2>/dev/null)
PUSHOVER_API_TOKEN=$(security find-generic-password -a pushover_token -s traffic_project -w 2>/dev/null)

PYTHON=/Users/jrill/Documents/traffic_project/venv311/bin/python3
SCRIPT=/Users/jrill/Documents/traffic_project/download_dav.py

WF_TMP=$(mktemp)
EF_TMP=$(mktemp)
$PYTHON $SCRIPT WF --quiet > "$WF_TMP" 2>&1 &
$PYTHON $SCRIPT EF --quiet > "$EF_TMP" 2>&1 &
wait
WF_STATS=$(cat "$WF_TMP"); rm -f "$WF_TMP"
EF_STATS=$(cat "$EF_TMP"); rm -f "$EF_TMP"

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
echo "══════════════════════════════════════════════"
echo "  $TIMESTAMP"
echo "  WF  $WF_STATS"
echo "  EF  $EF_STATS"
echo "══════════════════════════════════════════════"

PIPELINE_LOG=/Users/jrill/Documents/traffic_project/pipeline.log
PIPELINE_PID=/tmp/traffic_pipeline.pid
echo "══════════════════════════════════════════════" >> $PIPELINE_LOG
echo "  $TIMESTAMP  pipeline start" >> $PIPELINE_LOG
echo "══════════════════════════════════════════════" >> $PIPELINE_LOG

if [ -f "$PIPELINE_PID" ] && kill -0 "$(cat $PIPELINE_PID)" 2>/dev/null; then
    PIPELINE_SUMMARY="already running — skipped"
    echo "  pipeline: $PIPELINE_SUMMARY" >> $PIPELINE_LOG
else
    echo $$ > "$PIPELINE_PID"
    $PYTHON /Users/jrill/Documents/traffic_project/run_pipeline.py 2>&1 | grep -v "GMC failed\|lkpyramid\|Assertion failed.*prevPyr\|^$" >> $PIPELINE_LOG
    rm -f "$PIPELINE_PID"
    TODAY=$(date "+%Y_%m_%d")
    $PYTHON /Users/jrill/Documents/traffic_project/extract_vehicle_crops.py --date $TODAY >> $PIPELINE_LOG 2>&1
    $PYTHON /Users/jrill/Documents/traffic_project/extract_plate_crops.py --date $TODAY >> $PIPELINE_LOG 2>&1
    $PYTHON /Users/jrill/Documents/traffic_project/generate_dashboard.py >> $PIPELINE_LOG 2>&1
    PUSHOVER_USER_KEY=$PUSHOVER_USER_KEY PUSHOVER_API_TOKEN=$PUSHOVER_API_TOKEN \
        $PYTHON /Users/jrill/Documents/traffic_project/camera_spotcheck.py >> $PIPELINE_LOG 2>&1
    SWEEP_STATE=/Users/jrill/Documents/traffic_project/sweep_state.json
    if [ -f "$SWEEP_STATE" ]; then
        CAMERA_PASSWORD=$CAMERA_PASSWORD $PYTHON /Users/jrill/Documents/traffic_project/cam_sweep.py --step "$WF_STATS" >> $PIPELINE_LOG 2>&1
    fi
    PIPELINE_SUMMARY=$(grep "\[pipeline:summary\]" $PIPELINE_LOG | tail -1 | sed 's/\[pipeline:summary\] //')
    TIMING_SUMMARY=$(grep "\[timing:total\]" $PIPELINE_LOG | tail -1 | sed 's/\[timing:total\]  //')
fi
echo "  pipeline: $PIPELINE_SUMMARY"
if [ -n "$TIMING_SUMMARY" ]; then
    echo "  $TIMING_SUMMARY"
fi
