"""
test_rag_e2e.py — End-to-End RAG Benchmark: Embeddings, Retrieval, & Generation.
Run: python test_rag_e2e.py
"""
import os
import time
import logging
from dotenv import load_dotenv, find_dotenv

os.environ["STREAMLIT_LOG_LEVEL"] = "error"
logging.getLogger("streamlit").setLevel(logging.ERROR)

load_dotenv(find_dotenv(usecwd=True), override=True)

from modules.processor import process_pdf_to_memory
from modules.advicor import get_finance_advice, get_header_direct


def main():
    pdf_path = "data/statement.pdf" if os.path.exists("data/statement.pdf") else "data/temp_statement.pdf"
    if not os.path.exists(pdf_path):
        print(f"ERROR: Sample PDF not found at {pdf_path}")
        return

    print("=" * 72)
    print("  END-TO-END RAG EVALUATION (EMBEDDINGS + RETRIEVAL + GENERATION)")
    print("=" * 72)
    print(f"PDF File : {pdf_path}")

    # ── 1. Ingestion + Embeddings ────────────────────────────────────────────
    t0 = time.perf_counter()
    db, opening, closing, first_page, raw_docs = process_pdf_to_memory(pdf_path)
    ingest_ms = (time.perf_counter() - t0) * 1000

    if db is None:
        print(f"ERROR during PDF ingestion: {raw_docs}")
        return

    total_vec = len(db.get()["ids"])
    parent_count = len(getattr(db, "parent_map", {}))
    print(f"\n[1] EMBEDDINGS & VECTOR STORE")
    print(f"    Pages (Parents) : {parent_count}")
    print(f"    Total Vectors   : {total_vec}")
    print(f"    Opening Balance : {opening}")
    print(f"    Closing Balance : {closing}")
    print(f"    Ingestion Time  : {ingest_ms:.1f} ms ({ingest_ms/1000:.2f}s)")

    # ── 2. Retrieval + Generation (Chat Advisor / Groq) ──────────────────────
    queries = [
        "What is the name of the account holder and account number?",
        "What are the major debit or UPI expenses in this statement?",
        "What is the opening and closing balance?",
    ]

    print(f"\n[2] RETRIEVAL + GENERATION (Chat Advisor)")
    retrieval_times = []
    gen_times = []

    for i, q in enumerate(queries, 1):
        print(f"\n  --- Query {i}: {q} ---")
        t_ret = time.perf_counter()
        hits = db.similarity_search(q, k=4)
        ret_ms = (time.perf_counter() - t_ret) * 1000
        retrieval_times.append(ret_ms)

        hit_pids = [h.metadata.get("parent_id") for h in hits]
        print(f"  Retrieval Time : {ret_ms:.1f} ms | Matched Pages: {hit_pids}")
        if hits:
            preview = hits[0].page_content.replace("\n", " ")[:120]
            print(f"  Top Chunk      : {preview}...")

        t_gen = time.perf_counter()
        ans = get_finance_advice(q, db, first_page)
        gen_ms = (time.perf_counter() - t_gen) * 1000
        gen_times.append(gen_ms)

        ans_clean = ans.strip().replace("\n", " ")
        print(f"  Generation Time: {gen_ms:.1f} ms")
        print(f"  Answer         : {ans_clean[:300]}")

    # ── 3. Structured Header Extraction (Audit / Gemini) ─────────────────────
    print(f"\n[3] STRUCTURED EXTRACTION (Audit Header via Gemini)")
    t_hdr = time.perf_counter()
    try:
        acc = get_header_direct(first_page)
        hdr_ms = (time.perf_counter() - t_hdr) * 1000
        print(f"  Extraction Time: {hdr_ms:.1f} ms")
        print(f"  Customer Name  : {acc.customer_name}")
        print(f"  Account Number : {acc.account_number}")
        print(f"  IFSC / Branch  : {acc.ifsc_code} / {acc.branch_name}")
        print(f"  Period         : {acc.statement_period}")
    except Exception as e:
        print(f"  ERROR          : {e}")

    # ── 4. Summary ───────────────────────────────────────────────────────────
    avg_ret = sum(retrieval_times) / len(retrieval_times)
    avg_gen = sum(gen_times) / len(gen_times)
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  PDF Ingestion + Embeddings : {ingest_ms:.1f} ms")
    print(f"  Avg Retrieval Latency      : {avg_ret:.1f} ms")
    print(f"  Avg Generation Latency     : {avg_gen:.1f} ms")
    print("=" * 72)


if __name__ == "__main__":
    main()
