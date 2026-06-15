import sys
import os
import ollama
from google import genai


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


def prompt_model(llm_model: str, prompt: str) -> str:
    try:
        if llm_model in OLLAMA_MODELS:
            response = ollama.chat(
                model = llm_model,
                messages = [{"role": "user", "content": prompt}],
            )
            return response.message.content

        elif llm_model in GEMINI_MODELS:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                return "[Gemini Error] GEMINI_API_KEY environment variable not set."
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model = llm_model,
                contents = prompt,
            )
            return response.text

        else:
            return (f"[Error] Unknown model: '{llm_model}'. Supported models"
                    f": {sorted(OLLAMA_MODELS | GEMINI_MODELS)}")

    except Exception as code:
        return f"[{llm_model} Error] {code}"


def main():
    if len(sys.argv) != 3:
        print("Usage: python prompt_model.py <model> <prompt>")
        print(f"Ollama models : {sorted(OLLAMA_MODELS)}")
        print(f"Gemini models : {sorted(GEMINI_MODELS)}")
        sys.exit(1)

    model = sys.argv[1].strip()
    prompt = sys.argv[2].strip()
    if not model or not prompt:
        print("Error: <model> and <prompt> cannot be empty.")
        sys.exit(1)

    response = prompt_model(model, prompt)
    print("\n--- RESPONSE ---\n")
    print(f"{response}")


if __name__ == "__main__":
    main()