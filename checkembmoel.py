from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("KIMI_API_KEY")

models = NVIDIAEmbeddings(nvidia_api_key=api_key).available_models
for m in models:
    print(m)