import asyncio
import math
import json
import logging
import os
import sys
from pathlib import Path
from fastmcp import Client
from prompt_model import prompt_model
from fastmcp.client.transports import PythonStdioTransport

logging.basicConfig(
	level=logging.INFO,
	format="[%(asctime)s] | %(levelname)s | %(message)s",
	datefmt="%m/%d/%y %H:%M:%S",
)

# ---------------------------------------------------------------------------
# ─── GLOBAL CONFIGURATION ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

DEBUG = True
LOCAL_MODEL = False

# model passed to prompt_model()
OLLAMA_MODELS = [
	"llama3.1",
	"phi3",
	"deepseek-r1:1.5b",
	"gemma3:1b",
]

GEMINI_MODELS = [
	"gemini-3.1-flash-lite",
	"gemini-2.5-flash-lite",
	"gemini-2.5-flash",
	"gemini-3-flash-preview",
]

MODEL = OLLAMA_MODELS[0] if LOCAL_MODEL else GEMINI_MODELS[0]
DB_PATH = Path("data/jobs_d1.db") if DEBUG else Path("data/jobs.db")
RATE_LIMITS_TXT = Path("./rate_limits.txt")

TEMPERATURE = 0.95
TOP_P = 0.95

# Hypothetical local model rate limits (local models not in rate_limits.txt)
# Formula: batch_size = floor(LOCAL_TPM / AVG_TOKENS_PER_JOB)
LOCAL_RPM = 60
LOCAL_TPM = 250_000

MAX_RETRIES				= 3
BACKOFF_BASE_SECONDS	= 2.0        # seconds; doubles each retry

PROMPT_LINES = [
	"Extract the tech stack from each job description.",
	"Reply ONLY in this JSON format, one line per job, no other explanation:",
	"<source_id>: <tag1>, <tag2>, <tag3>",
	"",
	"Rules:",
	"- Tags must be specific tools, languages or frameworks (e.g. Python, React, MySQL).",
	# "- No generic terms (e.g. 'Programming Language', 'Database', 'Deployment').",
	"- No duplicates, no brackets, no markdown, must be comma-separated.",
	"- If the description is vague but hints at a common stack (e.g. 'web development' might imply JavaScript, HTML, CSS), make your best guess.",
	"- Even a vague hint is better than nothing.",
	# "- If nothing can be inferred, output: <source_id>: N/A",
	"",
	"Example:",
	"91397216: Python, SQL, MySQL, MariaDB, Tableau, A/B testing",
	"91347112: Java, Spring Boot, Docker, Kubernetes",
	"91765212: Excel, PowerPoint, Python, C, C++",
	"",
	"--- DATA STARTS HERE ---",
]

# ---------------------------------------------------------------------------
# ─── MAIN CLI ENTRY POINT ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
	if not DB_PATH.exists():
		logging.warning(f"Input path not found: {DB_PATH}")
		sys.exit(1)
	if not os.access(DB_PATH, os.R_OK):
		logging.warning(f"Input path not readable: {DB_PATH}")
		sys.exit(1)
	tag_data(DB_PATH)


# ---------------------------------------------------------------------------
# ─── CORE TAG_DATA ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def tag_data(db_url: str):
	try:
		asyncio.run(_tag_data_async(str(db_url)))
	except Exception as code:
		logging.error(f"Fatal error: {code}")


async def _tag_data_async(db_url: str):
	db_server = PythonStdioTransport("db_server.py", args=[db_url])
	async with Client(db_server) as mcp:
		b_idx = 0
		while True:
			batch_size, retry_delay = await compute_batch_params(mcp)

			untagged_result = await mcp.call_tool("fetch_untagged_jobs", {"batch_size": batch_size})
			batch: list[dict] = (
				json.loads(untagged_result.content[0].text) if untagged_result.content else []
			)
			if not batch:
				break

			expected_ids = [str(job["source_id"]) for job in batch]
			prompt = _build_prompt(batch, PROMPT_LINES)
			parsed: dict[str, str] = {}
			for attempt in range(1, MAX_RETRIES + 1):
				try:
					raw = prompt_model(MODEL, prompt, temperature=TEMPERATURE, top_p=TOP_P)
					if not raw:
						raise ValueError("Empty response from model")
					parsed = _parse_response(raw, expected_ids)
					if len(parsed) != len(batch):
						raise ValueError(
							"Mismatch between batch size and response")
					break

				except Exception as code:
					logging.error(f"[Batch {b_idx}] Attempt {attempt} failed: {code}"
						f"Retrying in {retry_delay:.1f}s [{attempt+1}/{MAX_RETRIES}]")
					if attempt < MAX_RETRIES:
						await asyncio.sleep(retry_delay
							* (BACKOFF_BASE_SECONDS ** (attempt - 1)))
					else:
						logging.error(f"[Batch {b_idx}] All {MAX_RETRIES} attempts "
							"failed — skipping batch.")

			for job in batch:
				sid   = str(job["source_id"])
				stack = parsed.get(sid, "")
				if not stack:
					continue
				ok = await mcp.call_tool("update_tech_stack", {"source_id": sid, "tech_stack": stack})
				if ok:
					logging.info(f"Analyzed Job {sid}: {stack}")
			b_idx += 1

		if b_idx == 0:
			logging.info("No data to tag")


# ---------------------------------------------------------------------------
# ─── BATCH SIZE & RETRY DELAY ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

async def compute_batch_params(mcp: Client) -> tuple[int, float]:
	# Derive (batch_size, retry_delay_seconds) from rate limits.
	#
	# batch_size  = floor(TPM / est_tokens_per_job) capped at 30
	# retry_delay = ceil(60 / RPM)
	# Falls back to LOCAL_RPM / LOCAL_TPM hypothetical limits for local models
	limits: dict[str, int] = _parse_rate_limits(RATE_LIMITS_TXT)
	m   = limits.get(MODEL, {})
	tpm = m.get("tpm", LOCAL_TPM)
	rpm = m.get("rpm", LOCAL_RPM)
	avg_desc_length = await mcp.call_tool("count_avg_desc_length", {})
	job_count = await mcp.call_tool("count_jobs", {})
	est_tokens_per_job = math.ceil((
		json.loads(avg_desc_length.content[0].text)
		if avg_desc_length else 1000) / 4 + 300
	)
	job_count = int(json.loads(job_count.content[0].text)) if job_count else 0

	batch_size = (math.ceil(job_count / 2) if job_count <= 30
		else min(tpm // est_tokens_per_job // rpm,
			math.ceil(job_count / (rpm // MAX_RETRIES))))
	retry_delay = math.ceil(60 / rpm)
	return batch_size, float(retry_delay)


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


if __name__ == "__main__":
	main()