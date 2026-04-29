import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib

load_dotenv()

# ── Local embedding model — runs on your CPU, zero API calls, ~5-10s ──────────
# This replaces MistralAIEmbeddings which made 30-40 API calls and took 2-3 min.
# all-MiniLM-L6-v2 is tiny (80MB), fast, and good enough for transaction search.
# @st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_file_hash(pdf_path):
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def extract_opening_balance(docs) -> float:
    full_text = " ".join([d.page_content for d in docs])
    pattern = re.search(
        r'Opening\s+Balance[\s\S]{0,60}?([\d,]{4,}\.?\d{0,2})',
        full_text, re.IGNORECASE
    )
    if pattern:
        try:
            val = float(pattern.group(1).replace(",", ""))
            if val > 100:
                return val
        except ValueError:
            pass
    # Fallback: first number > 1000 with decimals
    numbers = re.findall(r'\b(\d[\d,]*\.\d{2})\b', full_text)
    for n in numbers:
        val = float(n.replace(",", ""))
        if val > 1000:
            return val
    return 0.0


def extract_closing_balance(docs) -> float:
    last_page_text = docs[-1].page_content
    numbers = re.findall(r'\b(\d[\d,]*\.\d{2})\b', last_page_text)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    return 0.0


# @st.cache_resource(show_spinner=False)
def process_pdf_to_memory(pdf_path: str):
    """
    Load PDF → extract balances from raw text → chunk → embed locally → Chroma.

    SPEED FIX: HuggingFaceEmbeddings runs locally on CPU.
    No API calls during embedding = 10x faster than MistralAIEmbeddings.

    Returns: (vectorstore, opening_balance, closing_balance, first_page_text, docs)
    """
    loader = PyPDFLoader(pdf_path)
    docs   = loader.load()

    first_page_text = docs[0].page_content[:2500]
    opening_balance = extract_opening_balance(docs)
    closing_balance = extract_closing_balance(docs)

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=100
    ).split_documents(docs)

    embeddings = get_embedding_model()
    file_hash = get_file_hash(pdf_path)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"statement_{file_hash}"
    )

    return vector_db, opening_balance, closing_balance, first_page_text, docs