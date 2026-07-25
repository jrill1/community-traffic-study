#!/bin/bash
source /Users/jrill/.zprofile

cd /Users/jrill/Documents/traffic_project

PIPELINE_PID=/tmp/traffic_pipeline.pid

# Don't pull while pipeline is running — git replacing the db mid-write breaks SQLite
if [ -f "$PIPELINE_PID" ] && kill -0 "$(cat $PIPELINE_PID)" 2>/dev/null; then
    echo "$(date): pipeline running — skipping sync"
    exit 0
fi

cp traffic_events.db /tmp/traffic_sync_db.bak 2>/dev/null
cp traffic_events.tsv /tmp/traffic_sync_tsv.bak 2>/dev/null

git stash -u
git pull
git stash pop

cp /tmp/traffic_sync_db.bak traffic_events.db 2>/dev/null && chmod 644 traffic_events.db
cp /tmp/traffic_sync_tsv.bak traffic_events.tsv 2>/dev/null
