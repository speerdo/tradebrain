#!/bin/bash
# Nightly backup of the local TradeBrain Postgres (docs/LOCAL_POSTGRES_PLAN.md).
# Dumps to ~/tradebrain-backups, keeps the last 14 dailies, then mirrors the
# folder to Google Drive via rclone. Password comes from ~/.pgpass.
set -euo pipefail

DIR="${BACKUP_DIR:-$HOME/tradebrain-backups}"
REMOTE="${BACKUP_REMOTE:-gdrive:tradebrain-backups}"
KEEP=14
PG_BIN=/usr/lib/postgresql/17/bin

mkdir -p "$DIR"
out="$DIR/tradebrain-$(date +%F).dump"
"$PG_BIN/pg_dump" -h localhost -p 5433 -U tradebrain -d tradebrain -Fc -f "$out.tmp"
mv "$out.tmp" "$out"
"$PG_BIN/pg_restore" --list "$out" > /dev/null   # fails if the dump is unreadable
echo "wrote $out ($(du -h "$out" | cut -f1))"

ls -1t "$DIR"/tradebrain-*.dump | tail -n +$((KEEP + 1)) | xargs -r rm --

# Never deletes on the remote — dailies pruned locally stay in Drive, so do the
# neon-final-*.dump files.
rclone copy "$DIR" "$REMOTE" --include "*.dump"
echo "copied to $REMOTE"
