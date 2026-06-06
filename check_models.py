import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is required to list available models.")

    client = genai.Client(api_key=api_key)
    print("Models that support generateContent:")

    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        methods = getattr(model, "supported_generation_methods", None) or []
        supported = set(actions) | set(methods)
        if "generateContent" in supported:
            print(f"- {model.name}")


if __name__ == "__main__":
    main()
