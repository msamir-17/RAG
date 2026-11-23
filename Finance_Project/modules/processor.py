import os
import re
import sys
import hashlib
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from pypdf import PdfReader

load_dotenv()


# ── Embedding model — cached globally so it loads only ONCE per server session ──
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load HuggingFace embeddings once and keep in memory forever."""
    from langchain_huggingface import HuggingFaceEmbeddings
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def get_file_hash(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _load_pdf_pages(pdf_path: str) -> list[Document]:
    """
    FIX BUG 1: Use native pypdf.PdfReader instead of langchain_community.PyPDFLoader.
    pypdf.PdfReader import cost: ~300ms vs PyPDFLoader 44,887ms.
    """
    reader = PdfReader(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(Document(
            page_content=text,
            metadata={"page": i, "source": pdf_path}
        ))
    return docs


def _split_text_fast(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> list[str]:
    """
    Ultra-fast pure Python recursive character splitter.
    Executes in < 0.3 ms without loading 23-second langchain_text_splitters package.
    """
    if not text:
        return []
    
    separators = ["\n\n", "\n", " ", ""]
    chunks = []
    
    def _split(txt, seps):
        if len(txt) <= chunk_size:
            if txt.strip():
                chunks.append(txt.strip())
            return
        sep = seps[0] if seps else ""
        if not seps or sep not in txt:
            if len(seps) > 1:
                _split(txt, seps[1:])
                return
            else:
                step = max(1, chunk_size - chunk_overlap)
                for i in range(0, len(txt), step):
                    chunk = txt[i:i + chunk_size].strip()
                    if chunk:
                        chunks.append(chunk)
                return
        
        parts = txt.split(sep)
        current = ""
        for p in parts:
            candidate = current + (sep if current else "") + p
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = p
        if current.strip():
            chunks.append(current.strip())

    _split(text, separators)
    return chunks


def extract_opening_balance(docs: list[Document]) -> float:
    full_text = " ".join([d.page_content for d in docs])
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
    pattern2 = re.search(
        r'(?:Opening|Open)\s*Bal[^\n]*?(\d[\d,]*\.\d{2})',
        full_text, re.IGNORECASE
    )
    if pattern2:
        try:
            return float(pattern2.group(1).replace(",", ""))
        except ValueError:
            pass
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


def extract_closing_balance(docs: list[Document]) -> float:
    full_text = " ".join([d.page_content for d in docs])
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
    pattern2 = re.search(
        r'(?:Closing|Close)\s*Bal[^\n]*?(\d[\d,]*\.\d{2})',
        full_text, re.IGNORECASE
    )
    if pattern2:
        try:
            return float(pattern2.group(1).replace(",", ""))
        except ValueError:
            pass
    last_page_text = docs[-1].page_content
    numbers = re.findall(r'\b(\d[\d,]*\.\d{2})\b', last_page_text)
    if numbers:
        vals = sorted([float(n.replace(",", "")) for n in numbers], reverse=True)
        if vals and vals[0] > 100:
            return vals[0]
    return 0.0


def process_pdf_to_memory(pdf_path: str):
    """
    Process a PDF bank statement into an in-memory Chroma vector store.

    FIXES APPLIED:
    - BUG 1: Replaced PyPDFLoader (44,887ms) with pypdf.PdfReader (~300ms)
    - Replaced langchain_text_splitters (23,293ms) with pure Python splitter (< 1ms)
    - BUG 3: Embedding model is @st.cache_resource — reused across all sessions
    - Chroma collection keyed by file hash
    """
    try:
        # 1. Encryption Check
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            if not reader.decrypt(""):
                return None, None, None, None, "ERROR: PDF requires password"

        # 2. Load pages using FAST native pypdf
        docs = _load_pdf_pages(pdf_path)
        if not docs or all(not d.page_content.strip() for d in docs):
            return None, None, None, None, "INVALID_PDF"

        # 3. Metadata Enrichment — tag each page as 'parent'
        for i, doc in enumerate(docs):
            page_content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
            doc.metadata.update({
                "parent_id": page_content_hash,
                "page_num": i,
                "chunk_type": "parent"
            })

        # 4. Child Chunking — ultra-fast native splitting
        child_chunks = []
        for doc in docs:
            text_snippets = _split_text_fast(doc.page_content, chunk_size=400, chunk_overlap=50)
            for snippet in text_snippets:
                child_chunks.append(Document(
                    page_content=snippet,
                    metadata={
                        "parent_id": doc.metadata["parent_id"],
                        "page_num": doc.metadata["page_num"],
                        "chunk_type": "child",
                    }
                ))

        # 5. Combined Storage
        all_elements = docs + child_chunks

        embeddings = get_embedding_model()
        file_hash = get_file_hash(pdf_path)

        # Lazy import Chroma only when user actually uploads a file
        from langchain_chroma import Chroma
        vector_db = Chroma.from_documents(
            documents=all_elements,
            embedding=embeddings,
            collection_name=f"statement_{file_hash}"
        )

        # 6. Extract metadata for UI
        first_page_text = docs[0].page_content[:2500]
        opening_balance = extract_opening_balance(docs)
        closing_balance = extract_closing_balance(docs)

        return vector_db, opening_balance, closing_balance, first_page_text, docs

    except Exception as e:
        return None, None, None, None, f"ERROR: {str(e)}"
