from modules.processor import process_pdf_to_chroma
from modules.advicor import get_finance_advice
import os

# 1. Path to your statement
pdf_path = "data/statement.pdf"

print("--- Step 1: Processing PDF ---")
process_pdf_to_chroma(pdf_path)
print("Done! Data saved to storage/chroma_db\n")

print("--- Step 2: Asking Question ---")
query = "What were my total Swiggy expenses in March?"
answer = get_finance_advice(query)

print(f"Question: {query}")
print(f"AI Advisor: {answer}")