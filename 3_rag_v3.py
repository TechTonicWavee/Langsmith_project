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
os.environ['LANGCHAIN_PROJECT'] = 'RAG Example2 a'

PDF_PATH = "islr.pdf"

@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()

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

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size=750, chunk_overlap=150):
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vs = build_vectorstore(splits)
    return vs

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

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(pdf_path: str, question: str):
    vectorstore = setup_pipeline(pdf_path, chunk_size=750, chunk_overlap=150)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    chain = parallel | prompt | llm | StrOutputParser()
    return chain.invoke(question, config={"run_name": "pdf_rag_query"})

if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    while True:
        try:
            q = input("\nQ: ").strip()
            if not q:
                continue
            ans = setup_pipeline_and_query(PDF_PATH, q)
            print("\nA:", ans)
        except KeyboardInterrupt:
            print("\nExiting...")
            break