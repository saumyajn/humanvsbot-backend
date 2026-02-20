import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Here are the models your key can use for chat:")
print("-" * 40)

# Loop through all available models
for m in genai.list_models():
    # We only care about models that can generate text/chat
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
