import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import hashlib
from pypdf import PdfReader
load_dotenv()

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
    # 1. Look for explicit "Opening Balance" label with a number nearby
    pattern = re.search(
        r'Opening\s+Balance[\s\S]{0,80}?([\d,]+\.?\d{0,2})',
        full_text, re.IGNORECASE
    )
    if pattern:
        try:
            val = float(pattern.group(1).replace(",", ""))
            if val > 100:
                return val
        except ValueError:
            pass
    # 2. Look for "OPENING" near a balance column value
    pattern2 = re.search(
        r'(?:Opening|Open)\s*Bal[^\n]*?(\d[\d,]*\.\d{2})',
        full_text, re.IGNORECASE
    )
    if pattern2:
        try:
            return float(pattern2.group(1).replace(",", ""))
        except ValueError:
            pass
    # 3. Look for the first transaction row balance (S.No 1 balance)
    first_row = re.search(
        r'(?:1[\s|]+\d[\d/-]+\s+\d[\d/-]+\s+.+?)[\s]+(\d[\d,]*\.\d{2})[\s]*$',
        full_text, re.MULTILINE
    )
    if first_row:
        try:
            return float(first_row.group(1).replace(",", ""))
        except ValueError:
            pass
    return 0.0


def extract_closing_balance(docs) -> float:
    full_text = " ".join([d.page_content for d in docs])
    # 1. Look for explicit "Closing Balance" label with a number nearby
    pattern = re.search(
        r'Closing\s+Balance[\s\S]{0,80}?([\d,]+\.?\d{0,2})',
        full_text, re.IGNORECASE
    )
    if pattern:
        try:
            val = float(pattern.group(1).replace(",", ""))
            if val > 0:
                return val
        except ValueError:
            pass
    # 2. Look for "CLOSING" near a balance column value
    pattern2 = re.search(
        r'(?:Closing|Close)\s*Bal[^\n]*?(\d[\d,]*\.\d{2})',
        full_text, re.IGNORECASE
    )
    if pattern2:
        try:
            return float(pattern2.group(1).replace(",", ""))
        except ValueError:
            pass
    # 3. Fallback: last page, largest number with decimals (likely a balance)
    last_page_text = docs[-1].page_content
    numbers = re.findall(r'\b(\d[\d,]*\.\d{2})\b', last_page_text)
    if numbers:
        # Pick the largest number (balances are typically the largest values)
        vals = sorted([float(n.replace(",", "")) for n in numbers], reverse=True)
        if vals and vals[0] > 100:
            return vals[0]
    return 0.0


# @st.cache_resource(show_spinner=False)

def process_pdf_to_memory(pdf_path: str):
    try:
        # 1. Encryption Check (Unchanged)
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            if not reader.decrypt(""):
                return None, None, None, None, "ERROR: PDF requires password"
        
        # 2. Loading (Unchanged)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        if not docs or all(not d.page_content.strip() for d in docs):
            return None, None, None, None, "INVALID_PDF"

        # 3. Metadata Enrichment (New Phase 2 Layer)
        # We tag every page as a 'parent' so we can retrieve full context later
        for i, doc in enumerate(docs):
            page_content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
            doc.metadata.update({
                "parent_id": page_content_hash,
                "page_num": i,
                "chunk_type": "parent"
            })

        # 4. Child Chunking (New Phase 2 Layer)
        # Small chunks (400 chars) are better for finding exact words/amounts
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        child_chunks = child_splitter.split_documents(docs)
        for chunk in child_chunks:
            chunk.metadata["chunk_type"] = "child"

        # 5. Combined Storage
        # We store both the Full Page and the Snippets
        all_elements = docs + child_chunks
        
        embeddings = get_embedding_model()
        file_hash = get_file_hash(pdf_path)

        vector_db = Chroma.from_documents(
            documents=all_elements,
            embedding=embeddings,
            collection_name=f"statement_{file_hash}"
        )

        # 6. Extract Meta for UI (Unchanged)
        first_page_text = docs[0].page_content[:2500]
        opening_balance = extract_opening_balance(docs)
        closing_balance = extract_closing_balance(docs)

        print(f"DEBUG: Ingested {len(docs)} parents and {len(child_chunks)} children.")
        return vector_db, opening_balance, closing_balance, first_page_text, docs
    
    except Exception as e:
        print(f"❌ PDF Processing Error: {str(e)}")
        return None, None, None, None, f"ERROR: {str(e)}"

    # Verification print