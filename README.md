# Resume Skill Gap Analyzer — AI Analyzer (Week 2)
 
## Project Description

This project is the **Week 2** of a 3-week Resume Skill Gap Analyzer workshop. Using the cleaned data from Week 1, build the AI component of a skill gap detection pipeline that handles the business logic and decision-making of the application.

Week 2 builds an end-to-end AI pipeline that enriches a job listings database and performs resume skill gap analysis. The project is split across three modules:
 
- **Day 0 (Project Setup):** Configures the dual-model environment — local Ollama models for privacy and offline use, and Google Gemini models for cloud-powered performance. A unified `prompt_model.py` abstraction routes prompts to the correct backend.
- **Day 1–2 (Tagging):** Reads raw job descriptions from a SQLite database and uses an LLM to extract and populate a `tech_stack` column for each job listing, using rate-limit-aware batching.
- **Day 3–4 (Skill Gaps):** Reads a candidate's resume and compares it against the tagged job database to deterministically identify missing technical skills, returning a structured Pydantic result.
Together, the modules demonstrate practical LLM integration patterns: model routing, batch processing with retry logic, token optimisation, structured output, and prompt hardening.

###

## Setup Instructions

### Prerequisites

| Requirement | Version    |
|-------------|------------|
| Python      | **3.14.x** |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | latest |

> All Python dependencies are pinned to exact versions in `pyproject.toml`. Do not manually upgrade packages.

> Install `SQLite3 Editor` extension on `VS Code` for better database reading.

---

### 1. Clone the repository

```bash
git clone [github_repo_link] [folder_name]
cd [folder_name]
```

---

### 2. Install `uv` (if not already installed)

**macOS / Linux (Ubuntu / Debian):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Linux (Fedora / Red Hat):**
```bash
sudo dnf install uv -y
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
After installation, restart your terminal and verify:

```bash
uv --version
```

---

### 3. Install the correct Python version

`uv` can manage Python versions directly. Run:

```bash
uv python install 3.14
```

---

### 4. Create the virtual environment and install dependencies

```bash
uv sync
```

This reads `pyproject.toml`, pins all exact versions, and creates a `.venv` directory at the project root. You only need to run this once (or again after any dependency changes).

---

### 5. Create the environment file in the project directory

```bash
echo 'GEMINI_API_KEY="<YOUR_API_KEY>"' > ".env"

# Replace <YOUR_API_KEY> with your Gemini API key, without the <> brackets.
# Tip: Ctrl-Shift-V to paste on terminal.
```

This creates a `.env` file containing the Gemini API key as an environment variable. This file will be called by the dotenv python library during script execution, allowing the Gemini Client to access the variable. This file has been added into the gitignore list so it will not be uploaded to the remote repository.

If you accidentally uploaded it, make sure you run `git rm --cached .env` to remove it, add the file to `.gitignore`, then commit and  push the changes.

```bash
git rm --cached .env
echo ".env" >> .gitignore
git commit -m "chore: removing .env from remote repo"
git push
```
---

### 6. Activate the virtual environment (optional)

`uv run` (used in all commands below) automatically uses the `.venv` without manual activation. However, if you need to activate it for other tooling:

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**To deactivate when done:**
```bash
deactivate
```

<br/>

## Usage

All commands use `uv run` so they work consistently across platforms without needing manual venv activation.

### Run individual stages

```bash
# Stage 0 - LLM calling (run without arguments to see available models)
uv run prompt_data.py <model_name> <prompt>

# Stage 1 — Tech Stack Tagging based on job description
uv run tag_data.py

# Stage 2 — Analyze and compare resume with database to determine skill gaps
uv run find_skill_gaps.py

```
> Adjust the `DEBUG` and `LOCAL_MODEL` flag to change resume/database and LLM models respectively.
>- `DEBUG = True` to use test files with smaller sample size. `False` for bigger/eval files.
>- `LOCAL_MODEL = True` for local Ollama models. `False` for Gemini cloud models.

### Expected input

Place raw `.mhtml` files into the `0_source/` directory before running:

```
└─ data/
  ├─ *.db	← database samples consisting of job listings
  └─ *.txt	← resume samples
```

>You can inspect the final database with `SQLite3 editor` on `VS Code`.

### Code formatting

All Python code is formatted with `ruff` (version `0.15.*`):

```bash
uv run ruff format .
uv run ruff check .
```
<br/>

###

## API / Function Reference

### **prompt_model.py** — `prompt_model(model, prompt, temperature, top_p)`

**Purpose:** A unified interface to send a prompt to either a local Ollama model or a Google Gemini model, based on the model name provided. Handles unexpected errors without crashing.

| Parameter     | Type    | Description |
|---------------|---------|-------------|
| `model`       | `str`   | Model identifier. |
| `prompt`      | `str`   | The text prompt to send to the model. |
| `temperature` | `float` | Sampling temperature controlling output randomness. Lower values (e.g. `0.0`) produce more deterministic responses; higher values increase creativity. |
| `top_p`       | `float` | Nucleus sampling threshold. The model considers only the smallest set of tokens whose cumulative probability exceeds this value. Works in conjunction with `temperature`. |

**Returns:** `str` — the model's text response.

The function internally routes to the Ollama REST API (`localhost:11434`) for local models, and to the Google AI SDK (`google.genai`) for Gemini models. If the model name does not match any known identifier, or if an API/network error occurs, the function catches the exception, retries up to 2 times and returns a safe fallback None object rather than crashing.

**Available Models**
| Local            | Gemini                 |
|------------------|------------------------|
| llama3.1         | gemini-3.1-flash-lite  |
| phi3             | gemini-2.5-flash       |
| deepseek-r1:1.5b | gemini-2.5-flash-lite  |
| gemma3:1b        | gemini-3-flash-preview |

---

### **db_server.py** — MCP SQL Server

**Purpose:** A FastMCP server that exposes the SQLite database as a tool-callable service. Rather than having `tag_data.py` and `find_skill_gaps.py` execute SQL directly, they connect to this server as MCP clients and invoke its tool, decoupling database access from application logic.

**How it works:**
- The server is started as a separate process via stdio transport before either `tag_data.py` or `find_skill_gaps.py` runs.
- Clients connect using `fastmcp.Client("db_server.py")` and invoke different query functions to read from or write to the database.
- All SQL execution — reads, batch updates, and aggregations — is routed through this server rather than direct `sqlite3` calls in the calling modules.
- The server holds the database connection and path internally; callers never reference the file path directly when querying.

**Mainly used by:** `tag_data.py` (batch reads and `tech_stack` updates) and `find_skill_gaps.py` (aggregating `tech_stack` values across all job listings).

```
├─ tag_data.py          ← tagging + MCP client
├─ find_skill_gaps.py   ← skill gap analyzing + MCP client
├─ db_server.py         ← FastMCP server (all DB ops)
└─ sql/
  ├─ count_avg_desc_length.sql
  ├─ fetch_all_tagged_jobs.sql
  ├─ fetch_untagged.sql
  └─ update_tech_stack.sql
```

---

### **tag_data.py** — `tag_data(db_url)`

**Purpose:** Reads all rows in the `jobs` table that have no value in the `tech_stack` column, and uses an LLM to extract a comma-separated list of technologies from each job description, writing the result back to the database.

| Parameter | Type  | Description |
|-----------|-------|-------------|
| `db_url`  | `str` | File path to the SQLite database (e.g. `jobs_d1.db`). |

**Returns:**\
*(Mandatory)* `None`.\
*(Bonus -- did not attempt)* A tuple `(int, float)` representing total tokens used (input + output) and total time elapsed in milliseconds.

**Behaviour:**
- Rows are processed in batches. Batch size is derived from the model's documented rate limits (RPM and TPM from `rate_limits.txt`) to avoid exceeding quotas.
- Each batch is retried on failure (e.g. response/batch size mismatch, API error), with a calculated retry delay.
- For every successfully tagged row, the job ID and extracted tech stack are printed to standard output.
- If no untagged rows remain, the function logs `No data to tag` and exits cleanly.
- All exceptions (API errors, DB errors, parsing failures) are caught and logged without crashing the process.

**Core Workflow**
```
Calculate batch size (e.g. 20 jobs per batch) → reference rate_limits.txt for gemini
        ▼
Read jobs with no tech_stack of batch size
        ▼
For each batch → send descriptions to LLM → get back tech stacks
        ▼
Parse the LLM response → match each tech stack to its job ID
        ▼
Write results back to the database
        ▼
Log each result to stdout
```

**Output format (stdout):**
```bash
[Batch 0] Attempt 1 failed: Mismatch between batch size and response
Analyzed Job 91397216: SQL, Python, Java, Spring Framework/Spring Boot, ...
Analyzed Job 91347112: Java, PyTorch, TensorFlow, scikit-learn, Git, CI/CD
```

---

### **find_skill_gaps.py** — `find_skill_gaps(input_file_path, db_url)`

**Purpose:** Compares the technical skills in a candidate's resume against the aggregated `tech_stack` data from the jobs database to identify skills that appear in the job market but are absent from the resume.

| Parameter         | Type  | Description |
|-------------------|-------|-------------|
| `input_file_path` | `str` | Path to the resume text file (e.g. `resume.txt`). |
| `db_url`          | `str` | File path to the tagged SQLite database. |

**Returns:** `SkillGapResult` — a Pydantic `BaseModel` with the following shape:

```python
class SkillGapResult(BaseModel):
    gaps: list[str]   # Sorted, lowercase list of missing skills
    # (additional fields may be added as needed)
```

**Behaviour:**
- Reads and parses `input_file_path` to extract the candidate's technical skills.
- Reads the `tech_stack` column from the `jobs` table by batches using `source_id` in ascending order.
- Compares the two sets and aggregates all unique technologies present in the job database but absent from the resume across all listings.
- Results are **sorted alphabetically** and **converted to lowercase**.
- Skills containing `/` are split into individual entries, with the exception of `A/B testing` and `CI/CD`, which are treated as single atomic skills.
- Certifications and non-technical skills (leadership, management, etc.) are excluded from the gap analysis.
- Determinism is enforced — two consecutive runs on the same inputs must produce identical output. LLMs are used only where necessary; deterministic set-difference logic is preferred for the core comparison.
- All errors are caught gracefully. No stack traces are surfaced to the caller.

**Module interaction:** `find_skill_gaps` depends on the database produced by `tag_data`. The `tech_stack` column must be populated before skill gap analysis can be run. Both modules access the database exclusively through `db_server.py` via MCP rather than direct SQLite calls.

**Core Workflow**
```
Resume skills (LLM-extracted)   Job DB skills (raw) ◀───────┐
       ▼                              ▼                      │
   normalize()                   normalize()                 │
   - split on /                  - split on /                │
   - handle exceptions           - handle exceptions         │
   - lowercase                   - lowercase                 │
       ▼                              ▼                      │
resume_skills = set(...)        job_skills = set(...)        │
       └──────────────┬───────────────┘                      │
                      ▼                                      │
       batch = job_skills - resume_skills                    │
                      ▼                                      │
           gaps = gaps.union(batch) ──────(next batch)───────┘
                      ▼
              sorted(list(gaps))
```

**Output format (stdout):**
```bash
% uv run find_skill_gaps.py
gaps=['alibaba cloud', 'api integration or web automation', 'aws', 'aws deployment and maintenance', 'azure', 'c++', 'cloud logs', 'datastudio', 'excel', 'gcp', 'github actions', 'google cloud', 'grafana', 'linux development environments', 'mongodb', 'mysql', 'nginx', 'node.js', 'oracle', 'php', 'postgresql', 'power bi', 'powerbi', 'prometheus', 'restful api design and development', 'spring boot', 'spring framework', 'sql server', 'version control']
```

<br/>

###

## Data / Assumptions

### Data Sources

| Source | Description |
|--------|-------------|
| `jobs*.db` | SQLite database containing a `jobs` table with at minimum a job ID, description text, and a `tech_stack` column (nullable). |
| `resume*.txt` | Plain text file containing a candidate's extracted resume content. |
| `rate_limits.txt` | Plain text file recording RPM, TPM, and RPD for each Gemini model, used to calculate safe batch sizes and retry intervals. |

### Database Schema (`jobs` table — relevant columns)

| Column        | Type        | Description   |
|---------------|-------------|---------------|
| `source_id`   | TEXT        | Unique job listing identifier. |
| `job_tile`    | TEXT        | Job title.    |
| `company`     | TEXT        | Company name. |
| `description` | TEXT        | Raw job description text used for tagging. |
| `tech_stack`  | TEXT / NULL | Comma-separated tech stack populated by `tag_data`. NULL until tagged. |

### Assumptions

- The `jobs` table exists and is accessible at the provided `db_url`. No schema migration is performed.
- `resume.txt` is UTF-8 encoded plain text. No PDF or DOCX parsing is performed.
- Gemini API keys are available via environment variables (e.g. via `dotenv`). Ollama is running locally on `localhost:11434`.
- Batch sizes for `tag_data` are calculated from the rate limits in `rate_limits.txt` — specifically, batch size is bounded so that a full batch's estimated token usage does not exceed the model's TPM/RPM limits.
- Retry delay between failed batches is similarly derived from the rate limit figures rather than being hardcoded.
- Slight inaccuracy in tech stack extraction is acceptable. Determinism is **not** required for tagging.
- Determinism **is** required for skill gap results. The same resume and database should produce the same `gaps` list across runs.
- A skill written as `AWS/Azure/GCP` is treated as three separate skills: `aws`, `azure`, `gcp`. `A/B testing` and `CI/CD` are excluded from this splitting rule.
- Direct match accuracy is required for skill gap detection: if `C/C++` appears in the resume, the gap result must not list `c`, `c++`, or `c/c++` as missing — all three representations must be resolved before comparison.

### Data Flow

```
   jobs database (raw descriptions)
        ▼
   tag_data.py  ──► LLM (Gemini / Ollama)
        ▼
   jobs database (tech_stack populated)
        ├────────────────────────┐
        ▼                        ▼
   jobs.tech_stack           resume.txt
        └──────────┬─────────────┘
                   ▼
          find_skill_gaps.py  ──► LLM (Gemini / Ollama)
                   ▼
           SkillGapResult.gaps
```

###

<br/>

## Testing

### `prompt_model.py`

The module includes a `main()` function that calls `prompt_model` with each of the six supported model identifiers and prints the response. Manual verification confirms:
- Ollama local models respond correctly when the Ollama service is running.
- Gemini models respond correctly when a valid API key is configured.
- An invalid model name or a downed service returns a safe fallback None object rather than raising an exception.

### `tag_data.py`

**Test scenario 1 — Full run:**
Run `uv run tag_data.py` against a fresh database with untagged rows. Verify that:
- Each job ID appears in stdout with a comma-separated tech stack.
- The `tech_stack` column in the database is populated after the run.
- No Python stack traces appear.

**Test scenario 2 — No-op run:**
Run `uv run tag_data.py` a second time against the same database. Verify that:
- Stdout shows `No data to tag`.
- The database is unchanged.
- (bonus) Token count reported is 0.

**Test scenario 3 — Error injection:**
Temporarily revoke the API key or set an invalid `db_url`. Verify that the function logs the error and exits cleanly without a crash.

**Batch retry verification:**
Introduce a deliberate mismatch between batch size and API response by setting batch size to an extreme value. Confirm that `[Batch N] Attempt X failed: ...` messages appear and the function retries before succeeding or skipping.

### `find_skill_gaps.py`

**Determinism test:**
Run `uv run find_skill_gaps.py` twice consecutively on the same `resume.txt` and database. Compare the two `SkillGapResult.gaps` lists. They should be identical.

**Skill parsing test:**
Ensure a skill written as `AWS/Azure/GCP` in the database produces three separate entries (`aws`, `azure`, `gcp`) in the gap output if none appear in the resume.

**Exception test (`A/B testing` and `CI/CD`):**
Confirm that `A/B testing` and `CI/CD` are never split and appear as single skills in both parsing and output.

**Direct match test:**
Add `C/C++` to the resume. Confirm that none of `c`, `c++`, or `c/c++` appear in the gaps list regardless of how they were stored in the database.

**Non-technical filter test:**
Confirm that skills like `leadership` or `project management` do not appear in the gaps output even if present in job descriptions.

**Output format test:**
Verify `gaps` is sorted alphabetically and all entries are lowercase.

###

<br/>

## Limitations

### `prompt_model.py`
- No streaming support. Long responses from large models (e.g. `llama3.1`) may have high latency.
- Model selection is based on string matching of the model identifier. Typos or unofficial aliases will return a fallback.
- Local Ollama models may load slowly on first invocation; there is no warm-up mechanism.

### `tag_data.py`
- Tagging is non-deterministic — results may differ between runs for the same job description depending on LLM temperature and sampling.
- Batch size calculation depends on accurate entries in `rate_limits.txt`. Incorrect values may cause rate limit errors or unnecessarily slow processing.
- No deduplication of tech stack values within a single row. If the LLM repeats a skill, it appears multiple times.
- Large databases with thousands of untagged rows may still be slow due to API rate limits, even with optimised batching.
- (bonus) Token counting falls back to a 4-tokens-per-word estimate if the model does not return usage metadata. This is an approximation and may not reflect actual billing.

### `find_skill_gaps.py`
- Skill gap analysis is only as accurate as the tagging step. If `tag_data` misses or misspells a technology, the gap result will reflect that error.
- The determinism guarantee depends on the gap analysis logic being non-LLM or using a fixed seed/temperature. LLM calls within this module (if any) must be strictly constrained.
- Certifications (e.g. AWS Certified Solutions Architect) are intentionally excluded, which may cause the gaps list to miss skills that are implied by certifications.
- Resume parsing is limited to plain text. Resumes with complex formatting, tables, or multi-column layouts may not parse correctly if converted to `.txt` with information loss.

###

<br/>
 
## Architecture Reflection
 
### Design Choices
 
The project is deliberately split into three independent modules rather than a single monolithic script. `prompt_model.py` acts as a service abstraction layer — any upstream module calls it without knowing whether the request is going to a local Ollama instance or a cloud Gemini API.

Exposing `model`, `temperature`, and `top_p` as explicit parameters rather than hardcoding them reflects both a flexibility and a separation-of-concerns decision. The function's responsibility is to execute a prompt correctly — not to decide which model is appropriate or how creative the response should be. That judgement belongs to the caller. `tag_data.py` can freely choose a higher temperature since tagging does not need to be reproducible, while `find_skill_gaps.py` can pass `temperature=0.0` to satisfy the determinism requirement. At the same time, `prompt_model` still validates the `model` parameter against its list of supported identifiers and returns a safe fallback on an unrecognised value, so the flexibility given to callers does not compromise robustness.

`tag_data.py` and `find_skill_gaps.py` each own a single responsibility: enrichment and analysis respectively. The SQLite database acts as the shared state between them — a deliberate choice to keep the pipeline composable and re-runnable at any stage without reprocessing everything from scratch.

The use of Pydantic in `find_skill_gaps.py` enforces a typed contract on the output, making it easier to consume downstream (e.g. in a UI or another pipeline stage) and reducing the chance of silent schema drift.

### Trade-offs

The primary trade-off in `tag_data.py` is **speed vs. cost control**. Batch processing with rate-limit-derived sizes sacrifices raw throughput for predictable API usage and avoids unexpected billing spikes or cooldowns — an important constraint when working with free-tier Gemini quotas.

For `find_skill_gaps.py`, the trade-off is **determinism vs. LLM flexibility**. Skill gap detection relies on set-difference logic rather than asking an LLM to reason about gaps, because LLM outputs are non-deterministic by default. LLMs are only used where rule-based approaches cannot reasonably substitute — such as parsing freeform resume text — and even then, output is post-processed through normalisation rules to enforce consistency.

### Improvements

Given more time, the following extensions would strengthen the system:

- **Caching layer:** Cache LLM responses keyed on a hash of the input text. Re-runs on unchanged data would be near-instant and use zero tokens.
- **Skill normalisation database:** A lookup table mapping known aliases (`scikit-learn` / `sklearn`, `JS` / `JavaScript`) would reduce false positives in gap detection caused by naming inconsistencies between resumes and job listings.
- **Weighted gap scoring:** Rather than a binary present/absent gap list, scoring skills by how frequently they appear in the job database would prioritise high-demand gaps over niche ones.
- **Async batching:** `tag_data.py` currently processes batches sequentially. An async implementation with `asyncio` would allow concurrent API calls up to the rate limit, significantly reducing wall-clock time.
