# Move off Neon → local Postgres

**Goal:** stop paying Neon ~$1+/day for always-on compute. The agent runs on this
machine (`tradebrain.service`, systemd user unit) and touches the DB every 30 s
(position monitor) / 5 min (signal loop), so Neon never reaches its 5-minute idle
suspend. Moving the DB onto the same box removes that cost and a network hop.

**Approach:** run Postgres locally, copy the data over with `pg_dump`/`pg_restore`,
point `DATABASE_URL` at it. No query or schema changes — asyncpg, pgvector,
JSONB, Burt's Postgres-dialect SQL tool all keep working as-is.

## Status — cut over 2026-10-07 20:17 UTC

Done: Phases 1–4, Phase 5.1. Remaining: delete the Neon project after a clean
week (~2026-10-14). Deviations from the plan below, found during rehearsal:

- **Timezone + collation.** Neon runs `TimeZone=GMT`, `C.UTF-8`; the local
  cluster defaulted to `America/New_York`, `en_US.UTF-8`. That matters for
  `closed_at >= CURRENT_DATE` (daily-loss seed) and text `ORDER BY`. Fixed with
  a superuser-owned template DB `tradebrain_template` (`C.UTF-8`, `vector`
  installed, `IS_TEMPLATE`) and `ALTER DATABASE tradebrain SET timezone='UTC'`.
- **Recreating the DB without sudo.** `vector` is not a trusted extension, so
  `tradebrain` (granted `CREATEDB`) clones the template instead:
  `dropdb tradebrain && createdb -T tradebrain_template tradebrain`, then
  re-run the `ALTER DATABASE ... SET timezone` (per-DB settings don't copy).
- **Only expected restore error:** `must be owner of extension vector` on its
  `COMMENT` — harmless.
- **Password** lives in `~/.pgpass` only; `DATABASE_URL` has no password.
- **Backups:** `tradebrain-backup.timer` 03:30 daily → `~/tradebrain-backups`
  (14 kept) → `gdrive:tradebrain-backups` (never pruned). Restore from a Drive
  copy was tested. `neon-final-2026-10-07.dump` is in both places.
- **Rollback:** pre-cutover `.env` saved at `~/tradebrain-backups/.env.pre-local-pg`;
  the Neon line is also commented in `.env`.

## Facts this plan is built on (checked 2026-10-07)

| | |
|---|---|
| Neon server | PostgreSQL **17.11**, extensions `plpgsql`, `vector 0.8.0` |
| Neon size | 29 MB, 14 tables, no open trades |
| Local | Ubuntu 24.04; `postgresql-16` cluster `16/main` already online on **:5432**; pgvector **not** installed; Ubuntu repo only offers pgvector 0.6.0 |
| Local clients | `pg_dump`/`pg_restore` **16** — these refuse to dump a 17 server |
| Code | Only coupling to Neon is `DATABASE_URL` and the "Neon" wording in log lines/docs. `asyncpg.create_pool(dsn=...)` has no SSL/Neon options. Burt's read-only tool uses `transaction(readonly=True)` + `statement_timeout`, which is plain Postgres. |
| Live exits | Live entries place an exchange-native protective stop (`executor._place_protective_stop`), so a few minutes of agent downtime during cutover doesn't leave a live position unprotected. |

**Decision: install Postgres 17 from the PGDG apt repo as a separate cluster on
port 5433.** Matching Neon's major version makes the restore lossless, PGDG ships
a current `postgresql-17-pgvector` (0.8.x, same as Neon), and the existing 16
cluster is left untouched in case anything else on this machine uses it.
(Alternative: reuse the 16 cluster — saves a service but needs pg_dump 17 anyway
and a downgrade restore. Not worth it.)

---

## Phase 1 — Install and prepare local Postgres 17 (no impact on the live bot)

1. Add the PGDG repo and install:
   ```bash
   sudo apt install -y postgresql-common
   sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh   # adds PGDG repo
   sudo apt install -y postgresql-17 postgresql-17-pgvector
   ```
   `postgresql-common` will create cluster `17/main`, auto-assigned to port 5433
   because 5432 is taken. Confirm with `pg_lsclusters`.
   Watch for: apt may also offer to upgrade `postgresql-16` packages from PGDG —
   minor-version only, safe.

2. Create the role and database:
   ```bash
   sudo -u postgres psql -p 5433 <<'SQL'
   CREATE ROLE tradebrain LOGIN PASSWORD '<generate: openssl rand -base64 24>';
   CREATE DATABASE tradebrain OWNER tradebrain;
   \c tradebrain
   CREATE EXTENSION vector;
   SQL
   ```
   Creating the extension as superuser up front avoids needing superuser for the
   restore.

3. Confirm network exposure is local-only: `listen_addresses` defaults to
   `localhost` in `/etc/postgresql/17/main/postgresql.conf`; leave it. Default
   `pg_hba.conf` already allows `scram-sha-256` on 127.0.0.1.

4. Tuning: none needed at 29 MB. Defaults (128 MB shared_buffers) are plenty.

## Phase 2 — Rehearsal copy (bot keeps running on Neon)

1. Dump from Neon with the 17 client (PGDG put it at `/usr/lib/postgresql/17/bin`):
   ```bash
   /usr/lib/postgresql/17/bin/pg_dump "$NEON_URL" -Fc --no-owner --no-acl \
     -f ~/tradebrain-backups/neon-rehearsal.dump
   ```
2. Restore:
   ```bash
   /usr/lib/postgresql/17/bin/pg_restore -h localhost -p 5433 -U tradebrain \
     -d tradebrain --no-owner --no-acl --exit-on-error \
     ~/tradebrain-backups/neon-rehearsal.dump
   ```
   Expect one harmless error about `CREATE EXTENSION vector` already existing, so
   run without `--exit-on-error` if that trips it, then read the error list. Any
   other error is a stop-and-look.
3. Verify with a small script `scripts/compare_dbs.py` (new, committed). For each
   table in `public` it should check:
   - exact `count(*)` on both sides (not `pg_stat` — those numbers are stale)
   - `max(id)` and that each sequence's `last_value` ≥ `max(id)` (otherwise the
     next INSERT collides on the PK)
   - `\d` column list and index list match (`information_schema.columns`,
     `pg_indexes`)
   Read-only on Neon, so safe to run while live.
4. Smoke-test the app against the copy **without** touching the live service:
   `DATABASE_URL=<local> venv/bin/python scripts/test_quick.py` (or the DB-only
   checks in `scripts/live_preflight.py`). The agent must not be started a second
   time against live Coinbase — run checks only, not `agent.main`.

## Phase 3 — Cutover (~5 minutes of agent downtime)

Pick a time with no open position if possible (`SELECT * FROM trades WHERE
status='open'`). If one is open, it's still covered by its exchange stop.

1. `systemctl --user stop tradebrain` and confirm the process is gone
   (`pgrep -f agent.main`) so no more writes reach Neon.
2. Final dump from Neon → `~/tradebrain-backups/neon-final-<date>.dump`. **Keep
   this file permanently** — it's the last authoritative copy before the move.
3. Reset local and restore it:
   `dropdb`/`createdb` + `CREATE EXTENSION vector`, then `pg_restore` as in Phase 2.
4. Run `scripts/compare_dbs.py` again. Counts must match exactly.
5. Edit `.env`: keep the old line commented as a fallback.
   ```
   # DATABASE_URL=postgresql://...neon.tech/...   (pre-2026-10 Neon, rollback)
   DATABASE_URL=postgresql://tradebrain:<pw>@localhost:5433/tradebrain
   ```
6. Add `After=postgresql@17-main.service` to the unit? It's a **user** unit and
   can't order against system units, so instead rely on `Restart=always` +
   `RestartSec=10` (already set): if Postgres is slow at boot, the agent crash-
   loops a few times and comes up. Verify `StartLimitBurst=10 / 600s` gives
   enough headroom — it does (Postgres starts in seconds).
7. `systemctl --user start tradebrain`, then watch `logs/console.out` and
   `logs/agent.log` for: pool connected, open trades restored, first tick logs a
   signal row, first account snapshot written, position monitor running.
   Check the UI and ask Burt a SQL question to exercise `read_only_query`.

**Rollback:** stop the service, restore the commented Neon `DATABASE_URL`, start.
Anything written locally after cutover would be missing on Neon; if rollback
happens after real trades, dump local → restore to Neon first.

## Phase 4 — Backups (replaces what Neon gave us for free)

Trade/fill history is needed for taxes (Section 1256), so one disk isn't enough.

1. `scripts/backup_db.sh` (new, committed): `pg_dump -Fc` to
   `~/tradebrain-backups/tradebrain-YYYY-MM-DD.dump`, keep the last 14 dailies.
   Password via `~/.pgpass` (mode 600), not on the command line.
2. Schedule it as a systemd user timer (`tradebrain-backup.timer`, daily,
   `Persistent=true` so a missed run catches up after the machine was off).
3. Off-machine copy — **needs your choice of destination**: e.g. Google Drive
   via `rclone`, an external drive, or another machine via `rsync`. The dump is a
   few MB/day, so any of these is fine.
4. Test a restore from a backup file into a scratch DB once, so we know the
   backups actually work.

## Phase 5 — Clean up and shut down Neon

1. Code/docs wording, one commit:
   - `agent/database.py`: module docstring + "Connected to Neon DB" / "Closed
     Neon DB pool" log lines → "Postgres".
   - `scripts/setup_db.py` docstring, `.env.example` comment, `README.md`
     (setup table + architecture), `docs/BURT.md` pgvector section.
   - Historical plan docs (`BLUEPRINT.md`, `ACTION_PLAN.md`, etc.) stay as-is.
2. After **one week** of clean running and at least one verified backup: in the
   Neon console, delete the project (or at minimum set the compute to suspend so
   it costs ~$0). Keep the `neon-final-*.dump` file.
3. Update the memory note about where the DB lives.

## Risks

| Risk | Mitigation |
|---|---|
| Sequence values not carried over → PK collision on first insert | `pg_dump` includes `setval`; `compare_dbs.py` checks it explicitly |
| Writes land on Neon between final dump and env switch | Agent is stopped before the final dump |
| Disk/machine loss | Phase 4 off-machine backups; keep the final Neon dump |
| Machine off = DB off | Same as today: the agent already lives on this machine, so nothing new is lost |
| Port clash / wrong cluster | Explicit `-p 5433` everywhere; `pg_lsclusters` check in Phase 1 |

## Decisions needed from you

1. Off-machine backup destination (Phase 4.3).
2. OK to add the PGDG apt repo (needs `sudo`; you'll run those commands with `!`).
3. Is anything else using the existing Postgres 16 cluster? If not, it can be
   removed later; this plan doesn't touch it.
