"""
tag_data.py — Week 2 Day 1: Data Tagging
=========================================
Reads untagged job listings from a SQLite DB (via FastMCP), calls an LLM to
extract the tech stack from each job description, and writes results back.

Usage:
	uv run tag_data.py

To switch models, change the MODEL constant below.
"""

import asyncio
import math
import json
from pathlib import Path
from fastmcp import Client
from prompt_model import prompt_model
from fastmcp.client.transports import PythonStdioTransport

# ---------------------------------------------------------------------------
# ─── GLOBAL CONFIGURATION ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

DEBUG = True

# model passed to prompt_model()
OLLAMA_MODELS = [
	"gemma3:1b",
	"llama3.1",
	"phi3",
	"deepseek-r1:1.5b",
]

GEMINI_MODELS = [
	"gemini-3.1-flash-lite",
	"gemini-2.5-flash-lite",
	"gemini-2.5-flash",
	"gemini-3-flash-preview",
]

MODEL = OLLAMA_MODELS[1] if DEBUG else GEMINI_MODELS[0]
DB_PATH = Path("data/jobs_d1.db") if DEBUG else Path("data/jobs.db")
RATE_LIMITS_TXT = Path("./rate_limits.txt")

# Hypothetical local model rate limits (local models not in rate_limits.txt)
# Formula: batch_size = floor(LOCAL_TPM / AVG_TOKENS_PER_JOB)
LOCAL_RPM = 60
LOCAL_TPM = 50_000

MAX_RETRIES				= 3
BACKOFF_BASE_SECONDS	= 2.0        # seconds; doubles each retry


# ---------------------------------------------------------------------------
# ─── MAIN ENTRY POINT ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
	tag_data(DB_PATH)

# ---------------------------------------------------------------------------
# ─── CORE TAG_DATA ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def tag_data(db_url: str):
	try:
		asyncio.run(_tag_data_async(str(db_url)))
	except Exception as code:
		print(f"Fatal error: {code}")


async def _tag_data_async(db_url: str):
	server_cmd = PythonStdioTransport("db_server.py", args=[db_url])
	async with Client(server_cmd) as mcp:
		b_idx = 0
		while True:
			rate_limits: dict[str, int] = _parse_rate_limits(RATE_LIMITS_TXT)
			batch_size, retry_delay = await _compute_batch_params(rate_limits, mcp)
			untagged_result = await mcp.call_tool("fetch_untagged_jobs", {"batch_size": batch_size})
			batch: list[dict] = (
				json.loads(untagged_result.content[0].text) if untagged_result.content else []
			)
			if not batch:
				break

			prompt_lines = [
				"Extract the tech stack from each job description.",
				"Reply ONLY in this format, one line per job, no other text:",
				"<source_id>: <tag1>, <tag2>, <tag3>",
				"",
				"Rules:",
				"- Tags must be specific tools/languages/frameworks (e.g. Python, React, MySQL).",
				"- No generic terms (e.g. 'Programming Language', 'Database').",
				"- No duplicates, no brackets, no markdown.",
				"- If unsure, infer from job title and description.",
				"- If nothing can be inferred, output: <source_id>: N/A",
				"",
				"Example:",
				"91397216: Python, SQL, Tableau, A/B testing",
				"91347112: Java, Spring Boot, Docker, Kubernetes",
				"",
				"--- DATA STARTS HERE ---",
			]

			expected_ids = [str(job["source_id"]) for job in batch]
			prompt = _build_prompt(batch, prompt_lines)
			parsed: dict[str, str] = {}
			for attempt in range(1, MAX_RETRIES + 1):
				try:
					raw = await prompt_model(MODEL, prompt)
					print(raw + "\n")
					parsed = _parse_response(raw, expected_ids)
					if len(parsed) != len(batch):
						raise ValueError(
							"Mismatch between batch size and response")
					break

				except Exception as code:
					print(f"[Batch {b_idx}] Attempt {attempt} failed: {code}")
					if attempt < MAX_RETRIES:
						await asyncio.sleep(retry_delay
							* (BACKOFF_BASE_SECONDS ** (attempt - 1)))
					else:
						print(f"[Batch {b_idx}] All {MAX_RETRIES} attempts "
							"failed — skipping batch.")

			for job in batch:
				sid   = str(job["source_id"])
				stack = parsed.get(sid, "")
				if not stack:
					continue
				ok = await mcp.call_tool("update_tech_stack", {"source_id": sid, "tech_stack": stack})
				if ok:
					print(f"Analyzed Job {sid}: {stack}")
					b_idx += 1

		if b_idx == 0:
			print("No data to tag")


# ---------------------------------------------------------------------------
# ─── RATE LIMIT PARSING ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _parse_rate_limits(path: Path) -> dict[str, dict]:
	limits: dict[str, dict] = {}
	if not path.exists():
		return limits
	for line in path.read_text().splitlines():
		line = line.strip()
		if not line or line.startswith("#"):
			continue
		parts = line.split()
		if len(parts) < 4:
			continue
		model, rpm_s, tpm_s, rpd_s = parts[0], parts[1], parts[2], parts[3]
		limits[model] = {
			"rpm": _parse_num(rpm_s),
			"tpm": _parse_num(tpm_s),
			"rpd": _parse_num(rpd_s),
		}
	return limits


def _parse_num(s: str) -> int:
	s = s.upper().replace(",", "")
	if s.endswith("M"):
		return int(float(s[:-1]) * 1_000_000)
	if s.endswith("K"):
		return int(float(s[:-1]) * 1_000)
	return int(s)


# ---------------------------------------------------------------------------
# ─── BATCH SIZE & RETRY DELAY ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

async def _compute_batch_params(limits: dict, mcp: Client) -> tuple[int, float]:
	# Derive (batch_size, retry_delay_seconds) from rate limits.
	#
	# batch_size  = floor(TPM / est_tokens_per_job) capped at RPM and 20
	# retry_delay = ceil(60 / RPM)
	# Falls back to LOCAL_RPM / LOCAL_TPM hypothetical limits for local models
	m   = limits.get(MODEL, {})
	tpm = m.get("tpm", LOCAL_TPM)
	rpm = m.get("rpm", LOCAL_RPM)
	est_tokens = await mcp.call_tool("count_avg_desc_length", {})
	est_tokens = math.ceil((int(json.loads(est_tokens.content[0].text)
							if est_tokens else 1000) + 300) / 4)

	batch_size  = min(math.floor(tpm / est_tokens), rpm, 20)
	retry_delay = math.ceil(60 / rpm)
	return batch_size, float(retry_delay)


# ---------------------------------------------------------------------------
# ─── PROMPT BUILDER ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _build_prompt(jobs: list[dict], prompt_lines: list[str]) -> str:
	# Compact prompt — one line per job, no markdown or chain-of-thought.
	for job in jobs:
		desc = (job.get("description").replace("\n", " ") or "").strip()
		prompt_lines.append(f'[{job["source_id"]}] {job["job_title"]}'
					 f' @ {job["company"]}\nDescription:\n{desc}\n---')
	return "\n".join(prompt_lines)


# ---------------------------------------------------------------------------
# ─── RESPONSE PARSING ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _parse_response(raw: str, expected_ids: list[str]) -> dict[str, str]:
	# Parse LLM output into {source_id: tech_stack}.
	# Expects lines like:   <source_id>: tag1, tag2, tag3
	# Returns only the IDs we asked about; skips malformed lines.
	result: dict[str, str] = {}
	for line in raw.splitlines():
		line = line.strip()
		if not line or ":" not in line:
			continue
		sid, _, tags = line.partition(":")
		sid = sid.strip().strip("\"\'[]").strip()
		if sid in expected_ids:
			result[sid] = tags.strip().strip("\"\',[]").strip()
	return result


# ---------------------------------------------------------------------------
# ─── CLI ENTRY POINT ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

if __name__ == "__main__":
	main()