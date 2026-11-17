import io
import os

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def process_uploaded_file(uploaded_file) -> FAISS:
    """Extracts text from an uploaded PDF or TXT file and builds an in-memory FAISS index."""
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")

    if not text.strip():
        raise ValueError("No readable text found in the uploaded document.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)

    load_dotenv()
    embedding_model = os.getenv("EMBEDDING_MODEL", "").strip()
    if not embedding_model:
        raise ValueError("EMBEDDING_MODEL not found. Please add it to your .env file.")

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

def retrieve_context(vector_store: FAISS, query: str, k: int = 3) -> str:
    """Retrieves top-k relevant document chunks for a given query."""
    if vector_store is None:
        return ""
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])