import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def extract_opening_balance(docs) -> float:
    """
    Read opening balance directly from raw PDF text BEFORE chunking.
    Chunking splits pages and causes the AI to pick wrong balance values.
    This function reads the raw text and finds the opening balance reliably.
    """
    full_text = " ".join([d.page_content for d in docs])

    # Strategy 1: Find "Opening Balance" row and grab the number right after it
    pattern = re.search(
        r'Opening\s+Balance[\s\S]{0,60}?([\d,]{4,}\.?\d{0,2})',
        full_text, re.IGNORECASE
    )
    if pattern:
        raw = pattern.group(1).replace(",", "")
        try:
            val = float(raw)
            if val > 100:
                return val
        except ValueError:
            pass

    # Strategy 2: Grab the first number > 1000 (opening balance is always early)
    numbers = re.findall(r'\b(\d[\d,]*\.\d{2})\b', full_text)
    for n in numbers:
        val = float(n.replace(",", ""))
        if val > 1000:
            return val

    return 0.0


def extract_closing_balance(docs) -> float:
    """
    Read closing balance from the LAST page of the statement.
    The very last balance figure = closing balance.
    """
    last_page_text = docs[-1].page_content
    numbers = re.findall(r'\b(\d[\d,]*\.\d{2})\b', last_page_text)
    for n in reversed(numbers):
        val = float(n.replace(",", ""))
        if val > 1000:
            return val
    return 0.0


def process_pdf_to_memory(pdf_path: str):
    """
    Load PDF, extract key balance figures directly from raw text,
    then chunk + embed + store in in-memory Chroma.

    Returns:
        tuple: (vectorstore, opening_balance, closing_balance)
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # STRATEGY: Grab first page text for identity (first 2500 chars)
    first_page_text = docs[0].page_content[:2500] 

    # Existing logic for math and search
    opening_balance = extract_opening_balance(docs)
    closing_balance = extract_closing_balance(docs)

    chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100).split_documents(docs)
    vector_db = Chroma.from_documents(documents=chunks, embedding=MistralAIEmbeddings())

    return vector_db, opening_balance, closing_balance, first_page_text ,docs

def extract_customer_name(docs):
    full_text = " ".join([d.page_content for d in docs])
    # Look for "Customer Name" or "Name" and grab the next few words
    pattern = re.search(r'(?:Customer Name|Name)\s*:\s*([A-Z\s]{3,30})', full_text, re.IGNORECASE)
    if pattern:
        return pattern.group(1).strip()
    return "User"

