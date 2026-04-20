import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# We define a function so the UI can call it when a PDF is uploaded
def process_pdf_to_chroma(pdf_path, storage_directory="storage/chroma_db"):
    """
    1. Loads a PDF
    2. Splits into chunks
    3. Converts to Embeddings
    4. Saves into ChromaDB
    """
    
    # 1. Load the Document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 2. Chunking
    # We use Recursive splitter because it's smarter with paragraphs and sentences
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    
    # 3. Embeddings & Vector Store
    # This will create a folder inside Finance_Project/storage/
    embedding_model = MistralAIEmbeddings()
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=storage_directory
    )
    
    return vector_db