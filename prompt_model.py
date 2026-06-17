import sys
import ollama
from google import genai
from google.genai import types
from dotenv import load_dotenv

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

DEFAULT_TEMPERATURE = 0.95
DEFAULT_TOP_P = 0.95


def prompt_model(llm_model: str, prompt: str, temperature: float = DEFAULT_TEMPERATURE,
				 top_p: float = DEFAULT_TOP_P) -> str:
	llm_model = llm_model.strip()
	prompt = prompt.strip()
	if not llm_model or not prompt:
		print("Error: <model> and <prompt> cannot be empty.")
		return None

	try:
		if llm_model in OLLAMA_MODELS:
			response = ollama_client.generate(
				model = llm_model,
				prompt = prompt,
				options={
					"temperature": temperature,
					"top_p": top_p,
				},
			)
			return response.response

		elif llm_model in GEMINI_MODELS:
			response = gemini_client.models.generate_content(
				model = llm_model,
				contents = prompt,
				config = types.GenerateContentConfig(
					temperature = temperature,
					top_p = top_p,
				),
			)
			return response.text

		else:
			return (f"[Error] Unknown model: '{llm_model}'. Supported models"
					f": {sorted(OLLAMA_MODELS | GEMINI_MODELS)}")

	except Exception as code:
		return f"[{llm_model} Error] {code}"


async def main():
	if len(sys.argv) != 3:
		print("Usage: python prompt_model.py <model> <prompt>")
		print(f"Ollama models : {sorted(OLLAMA_MODELS)}")
		print(f"Gemini models : {sorted(GEMINI_MODELS)}")
		sys.exit(1)

	response = prompt_model(sys.argv[1], sys.argv[2])
	if response is not None:
		print("\n--- RESPONSE ---\n")
		print(f"{response}")

if __name__ == "__main__":
	main()