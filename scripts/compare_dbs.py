"""
Compare two Postgres databases table by table — used to verify the Neon →
local Postgres copy (docs/LOCAL_POSTGRES_PLAN.md). Read-only on both sides.

Checks per table in `public`: exact row count, max(id), that every sequence's
next value is past max(id) (else the next INSERT collides on the PK), and that
column and index definitions match.

Usage:
    venv/bin/python -m scripts.compare_dbs SOURCE_DSN TARGET_DSN

Exits 0 when everything matches, 1 otherwise.
"""
import asyncio
import sys

import asyncpg


async def describe(dsn: str) -> dict:
    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction(readonly=True):
            tables = [r["table_name"] for r in await conn.fetch(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
                "ORDER BY table_name"
            )]
            out = {"tables": {}, "sequences": {}, "extensions": {}}
            for t in tables:
                cols = [tuple(r) for r in await conn.fetch(
                    "SELECT column_name, data_type, udt_name, is_nullable, column_default "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = $1 "
                    "ORDER BY ordinal_position", t,
                )]
                idx = sorted(r["indexdef"] for r in await conn.fetch(
                    "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = $1", t,
                ))
                count = await conn.fetchval(f'SELECT count(*) FROM "{t}"')
                has_id = any(c[0] == "id" for c in cols)
                max_id = await conn.fetchval(f'SELECT max(id) FROM "{t}"') if has_id else None
                out["tables"][t] = {"cols": cols, "idx": idx, "count": count, "max_id": max_id}
            for r in await conn.fetch(
                "SELECT s.relname AS seq, t.relname AS tbl, a.attname AS col "
                "FROM pg_class s "
                "JOIN pg_depend d ON d.objid = s.oid AND d.deptype IN ('a', 'i') "
                "JOIN pg_class t ON t.oid = d.refobjid "
                "JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = d.refobjsubid "
                "WHERE s.relkind = 'S'"
            ):
                seq = await conn.fetchrow(f'SELECT last_value, is_called FROM "{r["seq"]}"')
                out["sequences"][r["seq"]] = {
                    "tbl": r["tbl"], "col": r["col"],
                    "next": seq["last_value"] + 1 if seq["is_called"] else seq["last_value"],
                }
            for r in await conn.fetch("SELECT extname, extversion FROM pg_extension"):
                out["extensions"][r["extname"]] = r["extversion"]
            return out
    finally:
        await conn.close()


async def main(src_dsn: str, dst_dsn: str) -> int:
    src, dst = await asyncio.gather(describe(src_dsn), describe(dst_dsn))
    problems = []

    print(f"extensions  source={src['extensions']}  target={dst['extensions']}")
    if "vector" in src["extensions"] and "vector" not in dst["extensions"]:
        problems.append("target is missing the vector extension")

    for t in sorted(set(src["tables"]) | set(dst["tables"])):
        s, d = src["tables"].get(t), dst["tables"].get(t)
        if s is None or d is None:
            problems.append(f"{t}: only on {'target' if s is None else 'source'}")
            continue
        ok = s["count"] == d["count"] and s["max_id"] == d["max_id"]
        print(f"{'ok ' if ok else 'BAD'} {t:24} rows {s['count']:>7} / {d['count']:<7} max_id {s['max_id']} / {d['max_id']}")
        if not ok:
            problems.append(f"{t}: row count or max(id) differs")
        if s["cols"] != d["cols"]:
            problems.append(f"{t}: column definitions differ")
        if s["idx"] != d["idx"]:
            problems.append(f"{t}: index definitions differ")

    for name, seq in sorted(dst["sequences"].items()):
        max_id = dst["tables"].get(seq["tbl"], {}).get("max_id")
        if max_id is not None and seq["col"] == "id" and seq["next"] <= max_id:
            problems.append(f"sequence {name}: next value {seq['next']} <= max(id) {max_id} on {seq['tbl']}")
        src_seq = src["sequences"].get(name)
        if src_seq and src_seq["next"] != seq["next"]:
            problems.append(f"sequence {name}: next {src_seq['next']} on source, {seq['next']} on target")

    if problems:
        print("\nMISMATCHES:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll tables, columns, indexes and sequences match.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1], sys.argv[2])))
