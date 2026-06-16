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
import json
import math
import sys
import time
from pathlib import Path
from fastmcp import Client
from prompt_model import prompt_model

# ---------------------------------------------------------------------------
# ─── GLOBAL CONFIGURATION ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

DEBUG = True
DB_NAME = Path("data/jobs_d1.db") if DEBUG else Path("data/jobs.db")

USE_LOCAL_MODEL = True

# model passed to prompt_model()
OLLAMA_MODELS = {
    "llama3.1",
    "phi3",
    "deepseek-r1:1.5b",
    "gemma3:1b",
}

GEMINI_MODELS = {
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
}

MODEL = OLLAMA_MODELS[3] if USE_LOCAL_MODEL else GEMINI_MODELS[0]

# Rate limits file (optional; if missing, falls back to hardcoded defaults)
RATE_LIMITS_TXT = Path("./rate_limits.txt")

# Hypothetical local model rate limits (local models not in rate_limits.txt)
# Formula: batch_size = floor(LOCAL_TPM / AVG_TOKENS_PER_JOB)
LOCAL_RPM = 60
LOCAL_TPM = 50_000

# Average token estimates
AVG_DESC_TOKENS    = 300    # input tokens per job description
AVG_RESPONSE_TOKENS = 20   # output tokens per job response line
AVG_TOKENS_PER_JOB = AVG_DESC_TOKENS + AVG_RESPONSE_TOKENS

# SQL files (referenced here for transparency; actually loaded in db_server.py)
SQL_FETCH_UNTAGGED    = Path("./sql/fetch_untagged.sql")
SQL_UPDATE_TECH_STACK = Path("./sql/update_tech_stack.sql")
SQL_FETCH_TAGGED      = Path("./sql/fetch_tagged.sql")

# Quality threshold
HIGH_QUALITY_MIN_TAGS = 3   # >= this many unique, non-duplicate tags → HIGH

# Max retries per batch and base back-off multiplier
MAX_RETRIES    = 3
BACKOFF_BASE_S = 2.0        # seconds; doubles each retry


# ---------------------------------------------------------------------------
# ─── CORE TAG_DATA ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def tag_data(db_url: str) -> tuple[int, int, float]:
    # Public entry point.
    # Returns (input_tokens, output_tokens, elapsed_ms).
    try:
        in_tok, out_tok, elapsed = asyncio.run(_tag_data_async(db_url))
        total_tok = in_tok + out_tok
        print(f"Total tokens used: {total_tok}, took {elapsed:.3f}ms")

    except Exception as exc:
        print(f"[tag_data] Fatal error: {exc}")

async def _tag_data_async(db_url: str) -> tuple[int, int, float]:
    rate_limits  = parse_rate_limits(RATE_LIMITS_TXT)
    batch_size, retry_delay = compute_batch_params(rate_limits)

    total_in  = 0
    total_out = 0
    start_ms  = time.monotonic()

    server_cmd = f"python db_server.py {db_url}"

    async with Client(server_cmd) as mcp:

        # ── 1. Fetch untagged jobs ──────────────────────────────────────────
        untagged_result = await mcp.call_tool("fetch_untagged_jobs", {})
        untagged: list[dict] = json.loads(untagged_result[0].text) if untagged_result else []

        if not untagged:
            print("No data to tag")
            print("High quality: 0     Low quality: 0")
            print("Empty/failed: 0     Duplicate rows: 0     Avg tags/job: 0")
            elapsed = (time.monotonic() - start_ms) * 1000
            return 0, 0, elapsed

        # ── 2. Process in batches ───────────────────────────────────────────
        batches = [untagged[i:i+batch_size] for i in range(0, len(untagged), batch_size)]

        for b_idx, batch in enumerate(batches):
            expected_ids = [str(j["source_id"]) for j in batch]
            prompt       = build_prompt(batch)
            parsed: dict[str, str] = {}

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    raw, in_tok, out_tok = await asyncio.to_thread(call_llm, prompt)
                    parsed = parse_response(raw, expected_ids)

                    if len(parsed) != len(batch):
                        raise ValueError(
                            f"Mismatch between batch size and response "
                            f"(expected {len(batch)}, got {len(parsed)})"
                        )

                    total_in  += in_tok
                    total_out += out_tok
                    break

                except Exception as exc:
                    print(f"[Batch {b_idx}] Attempt {attempt} failed: {exc}")
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(retry_delay * (BACKOFF_BASE_S ** (attempt - 1)))
                    else:
                        print(f"[Batch {b_idx}] All {MAX_RETRIES} attempts failed — skipping batch.")

            # ── 3. Write results back via MCP ───────────────────────────────
            for job in batch:
                sid   = str(job["source_id"])
                stack = parsed.get(sid, "")
                if not stack:
                    continue
                ok = await mcp.call_tool("update_tech_stack", {"source_id": sid, "tech_stack": stack})
                if ok:
                    print(f"Analyzed Job {sid}: {stack}")

        # ── 4. Fetch all tagged rows for quality report ─────────────────────
        all_tagged_result = await mcp.call_tool("fetch_tagged_jobs", {})
        all_tagged: list[dict] = json.loads(all_tagged_result[0].text) if all_tagged_result else []

        # ── 5. Quality report ───────────────────────────────────────────────
        report = compute_quality_report(all_tagged)
        print_quality_report(report)

    elapsed = (time.monotonic() - start_ms) * 1000
    return total_in, total_out, elapsed


# ---------------------------------------------------------------------------
# ─── RATE LIMIT PARSING ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def parse_rate_limits(path: Path) -> dict[str, dict]:
    # Parse rate_limits.txt.
    # Expected format (one model per line):
    #     <model_name> <RPM> <TPM> <RPD>
    # TPM values may use K/M suffixes (e.g. 250K).
    # Returns dict keyed by model name.
    limits: dict[str, dict] = {}
    if not path.exists():
        print(f"[warn] {path} not found — using hardcoded defaults.")
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

def compute_batch_params(limits: dict) -> tuple[int, float]:
    # Derive (batch_size, retry_delay_seconds) from rate limits.
    #
    # batch_size  = floor(TPM / avg_tokens_per_job)  capped at RPM and 20
    # retry_delay = ceil(60 / RPM)
    #
    # If MODEL is not found in rate_limits.txt (e.g. a local Ollama model),
    # falls back to LOCAL_RPM / LOCAL_TPM hypothetical limits.
    m   = limits.get(MODEL, {})
    tpm = m.get("tpm", LOCAL_TPM)
    rpm = m.get("rpm", LOCAL_RPM)

    batch_size  = max(1, min(math.floor(tpm / AVG_TOKENS_PER_JOB), rpm, 20))
    retry_delay = math.ceil(60 / rpm)
    return batch_size, float(retry_delay)


# ---------------------------------------------------------------------------
# ─── LLM CLIENTS ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> tuple[str, int, int]:
    # Call prompt_model and return (response_text, input_tokens, output_tokens).
    # Token counts are estimated since prompt_model returns text only.
    text       = prompt_model(MODEL, prompt)
    in_tokens  = _estimate_tokens(prompt)
    out_tokens = _estimate_tokens(text)
    return text, in_tokens, out_tokens


def _estimate_tokens(text: str) -> int:
    # Estimate token count at 4 tokens per word.
    # (prompt_model returns text only)
    return len(text.split()) * 4


# ---------------------------------------------------------------------------
# ─── PROMPT BUILDER ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def build_prompt(jobs: list[dict]) -> str:
    # Compact prompt — one line per job, no markdown or chain-of-thought.
    lines = [
        "Extract the technical stack from each job description.",
        "Rules: comma-separated technologies only, no explanations.",
        "Output exactly one line per job in the format:  <source_id>: <tag1>, <tag2>, ...",
        "---",
    ]
    for job in jobs:
        desc = (job.get("description") or "").strip()
        lines.append(f'[{job["source_id"]}]\n{job["job_title"]}'
                     f' @ {job["company"]}\nDescription:\n{desc}\n---')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ─── RESPONSE PARSING ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def parse_response(raw: str, expected_ids: list[str]) -> dict[str, str]:
    # Parse LLM output into {source_id: tech_stack}.
    # Expects lines like:   <source_id>: tag1, tag2, tag3
    # Returns only the IDs we asked about; skips malformed lines.
    result: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        sid, _, tags = line.partition(":")
        sid = sid.strip().lstrip("[").rstrip("]").strip()
        if sid in expected_ids:
            result[sid] = tags.strip()
    return result


# ---------------------------------------------------------------------------
# ─── QUALITY METRICS ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def classify_quality(tech_stack: str) -> str:
    # HIGH if ≥3 unique, non-empty, non-duplicate tags; else LOW.
    if not tech_stack or not tech_stack.strip():
        return "LOW"
    tags = [t.strip().lower() for t in tech_stack.split(",") if t.strip()]
    unique_tags = set(tags)
    has_duplicates = len(tags) != len(unique_tags)
    if len(unique_tags) >= HIGH_QUALITY_MIN_TAGS and not has_duplicates:
        return "HIGH"
    return "LOW"


def compute_quality_report(current: list[dict]) -> dict:
    # current: list of {source_id, tech_stack} dicts.
    # Returns a quality report dict.
    total       = len(current)
    empty_count = 0
    dup_count   = 0
    tag_counts  = []
    quality_labels: dict[str, str] = {}

    for row in current:
        sid   = row["source_id"]
        stack = row.get("tech_stack") or ""
        label = classify_quality(stack)
        quality_labels[sid] = label

        if not stack.strip():
            empty_count += 1
            continue

        tags = [t.strip().lower() for t in stack.split(",") if t.strip()]
        tag_counts.append(len(tags))
        if len(tags) != len(set(tags)):
            dup_count += 1

    avg_tags   = round(sum(tag_counts) / len(tag_counts), 2) if tag_counts else 0.0
    high_count = sum(1 for v in quality_labels.values() if v == "HIGH")
    low_count  = total - high_count

    return {
        "total_tagged":     total,
        "empty_failed":     empty_count,
        "duplicate_rows":   dup_count,
        "avg_tags_per_job": avg_tags,
        "high_quality":     high_count,
        "low_quality":      low_count,
        "quality_labels":   quality_labels,
    }


def print_quality_report(report: dict):
    print(f"High quality: {report['high_quality']}"
          f"  Low quality: {report['low_quality']}")
    print(f"Empty/failed: {report['empty_failed']}"
          f"  Duplicate rows: {report['duplicate_rows']}"
          f"  Avg tags/job: {report['avg_tags_per_job']}")


# ---------------------------------------------------------------------------
# ─── CLI ENTRY POINT ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    else:
        db_path = Path("jobs_d1.db")
    tag_data(db_path)