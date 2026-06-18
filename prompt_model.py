import sys
import logging
import time
import ollama
from google import genai
from google.genai import types
from dotenv import load_dotenv

logging.basicConfig(
	level=logging.INFO,
	format="[%(asctime)s] | %(levelname)s | %(message)s",
	datefmt="%m/%d/%y %H:%M:%S",
)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()
gemini_client = genai.Client()
ollama_client = ollama.Client()

OLLAMA_MODELS = {
	"llama3.1",
	"phi3",
	"deepseek-r1:1.5b",
	"gemma3:1b",
}

GEMINI_MODELS = {
	"gemini-3.1-flash-lite",
	"gemini-2.5-flash-lite",
	"gemini-2.5-flash",
	"gemini-3-flash-preview",
}

BASE_BACKOFF_SECONDS = 2.0
MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.95
DEFAULT_TOP_P = 0.95


def main():
	if len(sys.argv) != 3:
		print("Usage: python prompt_model.py <model> <prompt>")
		print(f"Ollama models : {sorted(OLLAMA_MODELS)}")
		print(f"Gemini models : {sorted(GEMINI_MODELS)}")
		sys.exit(1)

	response = prompt_model(sys.argv[1], sys.argv[2])
	if response is not None:
		print("\n--- RESPONSE ---\n")
		print(f"{response}")


def prompt_model(llm_model: str, prompt: str, temperature: float = DEFAULT_TEMPERATURE,
				 top_p: float = DEFAULT_TOP_P) -> str:
	# Basic validation and normalization
	llm_model = llm_model.strip() if llm_model else None
	prompt = prompt.strip() if prompt else None
	if not llm_model or not prompt:
		logging.error("<model> and <prompt> cannot be empty.")
		return None
	if llm_model not in OLLAMA_MODELS and llm_model not in GEMINI_MODELS:
		logging.error(f"Unknown model: '{llm_model}'. Supported models"
			f": {sorted(OLLAMA_MODELS | GEMINI_MODELS)}")
		return None

	temperature = max(0.0, min(1.0, temperature))
	top_p = max(0.0, min(1.0, top_p))

	try:
		if llm_model in OLLAMA_MODELS:
			for i in range(MAX_RETRIES):
				try:
					response = ollama_client.generate(
						model = llm_model,
						prompt = prompt,
						options={
							"temperature": temperature,
							"top_p": top_p,
						},
					)
					return response.response

				except ollama.ResponseError as e:
					if i < MAX_RETRIES - 1:
						delay = BASE_BACKOFF_SECONDS * (2 ** i)
						logging.warning(
							f"[{llm_model}]: {e.status_code} - {e.error}. "
							f"Retrying in {delay:.1f}s [{i+1}/{MAX_RETRIES}]")
						time.sleep(delay)
						continue
					raise ValueError(f"Error ({e.status_code}): {e.error}")

		if llm_model in GEMINI_MODELS:
			for i in range(MAX_RETRIES):
				try:
					response = gemini_client.models.generate_content(
						model = llm_model,
						contents = prompt,
						config = types.GenerateContentConfig(
							temperature = temperature,
							top_p = top_p,
						),
					)
					return response.text
				
				except genai.errors.APIError as e:
					if i < MAX_RETRIES - 1:
						delay = BASE_BACKOFF_SECONDS * (2 ** i)
						logging.warning(
							f"[{llm_model}]: {e.code} - {e.message}. "
							f"Retrying in {delay:.1f}s [{i+1}/{MAX_RETRIES}]")
						time.sleep(delay)
						continue
					raise ValueError(f"Error ({e.code}): {e.message}")

	except Exception as code:
		logging.error(f"[{llm_model}]: {code}")
		return None


if __name__ == "__main__":
	main()