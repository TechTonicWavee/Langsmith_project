import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("KIMI_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = 'RAG Example'

PDF_PATH = "islr.pdf"

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index
emb = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-1b-v2",
    api_key=api_key
)

vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) LLM
llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=api_key
)

# 6) Chain
def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 7) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
while True:
    q = input("\nQ: ")
    if not q.strip():
        continue
    ans = chain.invoke(q.strip())
    print("\nA:", ans)