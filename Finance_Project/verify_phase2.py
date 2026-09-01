import os
from modules.processor import process_pdf_to_memory

# 1. Path to your sample PDF
pdf_path = "data/temp_statement.pdf"

if not os.path.exists(pdf_path):
    print(f"❌ Error: {pdf_path} not found. Please upload a file via Streamlit first.")
else:
    print("--- Testing Parent-Child Ingestion ---")
    
    # 2. Run the new processor logic
    db, opening, closing, first_page, raw_docs = process_pdf_to_memory(pdf_path)
    
    # 3. Peek into the Vector Store
    # We retrieve a few items to check their metadata
    collection_data = db.get(limit=10)
    
    parents = [m for m in collection_data['metadatas'] if m.get('chunk_type') == 'parent']
    children = [m for m in collection_data['metadatas'] if m.get('chunk_type') == 'child']
    
    print(f"✅ Database contains {len(collection_data['metadatas'])} total chunks.")
    print(f"✅ Metadata Check - Parents found: {len(parents)}")
    print(f"✅ Metadata Check - Children found: {len(children)}")
    
    if len(parents) > 0 and len(children) > 0:
        print("\n🚀 SUCCESS: Parent-Child chunking is working correctly!")
        print("Child Chunk Sample Metadata:", children[0])
    else:
        print("\n❌ FAILURE: Metadata 'chunk_type' is missing or incorrect.")