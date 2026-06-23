from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")
api_key = os.getenv("KIMI_API_KEY")

model = ChatNVIDIA(api_key=api_key)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)
