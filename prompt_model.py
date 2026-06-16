import sys
import asyncio
import ollama
from google import genai
from fastmcp import Client
from dotenv import load_dotenv

load_dotenv()
mcp_client = Client("db_server.py")
gemini = genai.Client()

OLLAMA_MODELS = {
    "gemma3:1b",
    "llama3.1",
    "phi3",
    "deepseek-r1:1.5b",
}

GEMINI_MODELS = {
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
}


async def prompt_model(llm_model: str, prompt: str) -> str:
    llm_model = llm_model.strip()
    prompt = prompt.strip()
    if not llm_model or not prompt:
        print("Error: <model> and <prompt> cannot be empty.")
        return None

    try:
        if llm_model in OLLAMA_MODELS:
            response = await ollama.generate(
                model = llm_model,
                prompt = prompt,
            )
            return response.response

        elif llm_model in GEMINI_MODELS:
            response = await gemini.aio.models.generate_content(
                model = llm_model,
                contents = prompt,
                config = genai.types.GenerateContentConfig(tools=[mcp_client.session])
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

    response = await prompt_model(sys.argv[1], sys.argv[2])
    if response is not None:
        print("\n--- RESPONSE ---\n")
        print(f"{response}")

if __name__ == "__main__":
    asyncio.run(main())