from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

models = ChatNVIDIA(nvidia_api_key=api_key).available_models
for m in models:
    if not m.deprecated:
        print(m.id)