import asyncio
import logging
import math
import json
from pathlib import Path
from fastmcp import Client
from pydantic import BaseModel
from prompt_model import prompt_model
from process_resume import extract_resume_skills
from tag_data import compute_batch_params
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
	resume_skills = extract_resume_skills(Path(input_file_path))

	# Placeholder implementation - replace with actual skill gap finding logic
	return SkillGapResult(gaps=list(resume_skills))
