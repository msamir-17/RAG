import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
import shutil 
load_dotenv()




def process_pdf_to_memory(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_documents(docs)
    
    # Notice: No persist_directory here. It stays in RAM.
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=MistralAIEmbeddings()
    )
    return vector_db

# We define a function so the UI can call it when a PDF is uploaded
# def process_pdf_to_chroma(pdf_path, storage_directory="storage/default"):
    # --- AUTOMATIC CLEANUP ---
    # If the folder exists, delete it so we don't get duplicate data
    if os.path.exists(storage_directory):
        shutil.rmtree(storage_directory)
        print(f"Cleared old data in {storage_directory}")



    # 1. Load the Document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 2. Chunking
    # We use Recursive splitter because it's smarter with paragraphs and sentences
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=250
    )
    chunks = splitter.split_documents(docs)
    
    # 3. Embeddings & Vector Store
    # This will create a folder inside Finance_Project/storage/
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=storage_directory
    )
    
    return vector_db