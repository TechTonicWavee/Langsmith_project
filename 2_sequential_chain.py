import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_PROJECT'] = 'Sequential Chain Example'
load_dotenv()

api_key = os.getenv("KIMI_API_KEY")

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatNVIDIA(
    model="deepseek-ai/deepseek-r1",
    nvidia_api_key=api_key
)

model2 = ChatNVIDIA(
    model="qwen/qwen3-235b-a22b",
    nvidia_api_key=api_key
)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | (lambda text: {"text": text}) | prompt2 | model2 | parser

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)