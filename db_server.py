"""
db_server.py — FastMCP server for all SQLite DB operations.
All queries are loaded from ./sql/*.sql files.
Run standalone or used as a stdio MCP server by tag_data.py.
"""

import sqlite3
import sys
from pathlib import Path
from fastmcp import FastMCP

DB_PATH: str = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/jobs_d1.db")

SQL_COUNT_AVG_DESC_LEN  = Path("./sql/count_avg_desc_length.sql")
SQL_FETCH_UNTAGGED      = Path("./sql/fetch_untagged.sql")
SQL_UPDATE_TECH_STACK   = Path("./sql/update_tech_stack.sql")
SQL_FETCH_TAGGED_JOBS = Path("./sql/fetch_tagged_jobs.sql")
SQL_COUNT_JOBS = Path("./sql/count_jobs.sql")

# MCP server
mcp = FastMCP("SQLite-Service")


def _load_sql(path: Path) -> str:
	return path.read_text(encoding="utf-8").strip()


def _connect() -> sqlite3.Connection:
	conn = sqlite3.connect(DB_PATH)
	conn.row_factory = sqlite3.Row
	return conn


@mcp.tool()
def count_jobs() -> int:
	# Return the total number of jobs in the database.
	sql = _load_sql(SQL_COUNT_JOBS)
	with _connect() as conn:
		result = conn.execute(sql).fetchone()
	return int(result[0]) if result else 0


@mcp.tool()
def count_avg_desc_length() -> float:
	# Return the average length of job descriptions.
	sql = _load_sql(SQL_COUNT_AVG_DESC_LEN)
	with _connect() as conn:
		result = conn.execute(sql).fetchone()
	return float(result[0]) if result else 0


@mcp.tool()
def fetch_untagged_jobs(batch_size: int) -> list[dict]:
	# Return jobs where tech_stack is NULL or empty in batch.
	sql = _load_sql(SQL_FETCH_UNTAGGED)
	with _connect() as conn:
		rows = conn.execute(sql, {"batch_size": batch_size}).fetchall()
	return [dict(r) for r in rows]


@mcp.tool()
def fetch_tagged_jobs(batch_size: int, last_sid: int) -> list[dict]:
	# Return jobs where tech_stack is not null or empty in batch.
	sql = _load_sql(SQL_FETCH_TAGGED_JOBS)
	with _connect() as conn:
		rows = conn.execute(sql,
			{"batch_size": batch_size, "last_sid": last_sid}).fetchall()
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


if __name__ == "__main__":
	mcp.run()