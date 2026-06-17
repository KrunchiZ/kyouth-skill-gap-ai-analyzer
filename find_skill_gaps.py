import asyncio
import logging
import math
import json
from pathlib import Path
from fastmcp import Client
from pydantic import BaseModel
from prompt_model import prompt_model
from fastmcp.client.transports import PythonStdioTransport

class SkillGapResult(BaseModel):
	gaps: list[str]

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

# Hypothetical local model rate limits (local models not in rate_limits.txt)
# Formula: batch_size = floor(LOCAL_TPM / AVG_TOKENS_PER_JOB)
LOCAL_RPM = 60
LOCAL_TPM = 50_000

MAX_RETRIES				= 3
BACKOFF_BASE_SECONDS	= 2.0

SYSTEM_PROMPT = """\
"""


def find_skill_gaps(input_file_path: str, db_url: str) -> SkillGapResult:
	pass


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