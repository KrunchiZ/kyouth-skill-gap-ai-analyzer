"""
db_server.py — FastMCP server for all SQLite DB operations.
All queries are loaded from ./sql/*.sql files.
Run standalone or used as a stdio MCP server by tag_data.py.
"""

import sqlite3
import sys
from pathlib import Path
from fastmcp import FastMCP

DEBUG = True
DB_NAME = Path("data/jobs_d1.db") if DEBUG else Path("data/jobs.db")

SQL_FETCH_UNTAGGED    = Path("./sql/fetch_untagged.sql")
SQL_UPDATE_TECH_STACK = Path("./sql/update_tech_stack.sql")
SQL_FETCH_TAGGED      = Path("./sql/fetch_tagged.sql")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP("SQLite-Jobs-Service")


def _load_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def fetch_untagged_jobs() -> list[dict]:
    # Return all jobs where tech_stack is NULL or empty.
    sql = _load_sql(SQL_FETCH_UNTAGGED)
    with _connect() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


@mcp.tool()
def update_tech_stack(source_id: str, tech_stack: str) -> bool:
    # Write the extracted tech_stack for a single job row.
    sql = _load_sql(SQL_UPDATE_TECH_STACK)
    try:
        with _connect() as conn:
            conn.execute(sql, {"tech_stack": tech_stack, "source_id": source_id})
            conn.commit()
        return True
    except Exception as exc:
        print(f"[db_server] update_tech_stack error for {source_id}: {exc}", file=sys.stderr)
        return False


@mcp.tool()
def fetch_tagged_jobs() -> list[dict]:
    # Return all jobs that already have a tech_stack value (for quality checks).
    sql = _load_sql(SQL_FETCH_TAGGED)
    with _connect() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    mcp.run()