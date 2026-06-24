import os
from dotenv import load_dotenv
from langsmith import traceable
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

# ---------- traced setup steps ----------
@traceable(name="load_pdf")
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size=750, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    emb = NVIDIAEmbeddings(
        model="nvidia/llama-nemotron-embed-1b-v2",
        api_key=api_key
    )
    return FAISS.from_documents(splits, emb)

@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs)
    vs = build_vectorstore(splits)
    return vs

# ---------- pipeline ----------
llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    api_key=api_key
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer ONLY from the provided context. 
If the question is not related to the context, say 'This question is outside the scope of the document.'
Do not use any outside knowledge."""),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

vectorstore = setup_pipeline(PDF_PATH)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# ---------- run queries ----------
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
while True:
    try:
        q = input("\nQ: ").strip()
        if not q:
            continue
        config = {"run_name": "pdf_rag_query"}
        ans = chain.invoke(q, config=config)
        print("\nA:", ans)
    except KeyboardInterrupt:
        print("\nExiting...")
        break