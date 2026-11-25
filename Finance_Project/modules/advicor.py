import time
import os
import pandas as pd
import asyncio
import plotly.io as pio
from io import BytesIO
from typing import List, Optional
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import streamlit as st
from modules.schema import FullStatementReport, AccountDetails, Transaction, ForecastInsight
import datetime
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from modules.category_classifier import classify_transaction   # local sub-ms classifier
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from reportlab.platypus.flowables import Flowable

# ── Hybrid LLM Helpers: Google Gemini (Audit/Forecast) + Groq (Chat/Voice) ───
# Lazy-imported so cold start stays < 500ms.

_CACHED_GROQ_MODEL: Optional[str] = None
_GEMINI_MODELS = ("gemini-2.5-flash", "gemini-2.0-flash")


def _get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not found in .env")
    from google import genai
    return genai.Client(api_key=api_key.strip())


def _gemini_generate(prompt_text: str, schema=None, temperature: float = 0):
    """Call Google Gemini with optional Pydantic structured output and model fallback."""
    from google.genai import types
    client = _get_gemini_client()

    config_kwargs = {"temperature": temperature}
    if schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = schema
    config = types.GenerateContentConfig(**config_kwargs)

    last_err = None
    for model_name in _GEMINI_MODELS:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=config,
            )
            if schema is not None:
                if getattr(resp, "parsed", None) is not None:
                    return resp.parsed
                return schema.model_validate_json(resp.text)
            return resp.text or ""
        except Exception as e:
            last_err = e
            err_s = str(e).lower()
            if "404" in err_s or "not found" in err_s:
                continue
            raise
    raise last_err


def _pick_groq_chat_model(client) -> str:
    """Dynamically pick an available text-generation model from the user's Groq account."""
    global _CACHED_GROQ_MODEL
    if _CACHED_GROQ_MODEL:
        return _CACHED_GROQ_MODEL

    preferred = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]
    try:
        available = [
            m.id for m in client.models.list().data
            if not any(bad in m.id.lower() for bad in ["whisper", "guard", "tts", "playai", "distil-whisper"])
        ]
        for p in preferred:
            if p in available:
                _CACHED_GROQ_MODEL = p
                return p
        if available:
            _CACHED_GROQ_MODEL = available[0]
            return available[0]
    except Exception:
        pass
    return "openai/gpt-oss-120b"


def _groq_chat(prompt_text: str, temperature: float = 0, json_mode: bool = False) -> str:
    """Call Groq chat completions with dynamic model resolution."""
    global _CACHED_GROQ_MODEL
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env")
    client = Groq(api_key=api_key.strip())

    model_id = _CACHED_GROQ_MODEL or _pick_groq_chat_model(client)
    for attempt in range(2):
        try:
            kwargs = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            _CACHED_GROQ_MODEL = model_id
            return resp.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e).lower()
            if ("404" in err_str or "model_not_found" in err_str or "does not exist" in err_str) and attempt == 0:
                _CACHED_GROQ_MODEL = None
                model_id = _pick_groq_chat_model(client)
                continue
            raise
    return ""




class TransactionBatch(BaseModel):
    transactions: List[Transaction]


# ── LLM helpers ───────────────────────────────────────────────────────────────

def get_finance_advice(user_query: str, vectorstore, first_page_text: str = "") -> str:
    """
    Generate financial advice/answers using RAG.
    Includes automatic Groq fallback on Mistral 429 Rate Limit and compact context assembly.
    """
    # 1. Similarity search on the vectorstore
    hits = vectorstore.similarity_search(user_query, k=4)

    # 2. Extract parent IDs from hits
    parent_map = getattr(vectorstore, "parent_map", {})
    parent_ids = list(dict.fromkeys([h.metadata.get("parent_id") for h in hits if h.metadata.get("parent_id")]))

    # 3. If query asks about identity / account info, guarantee first page (p_0) is present
    q_lower = user_query.lower()
    is_account_query = any(k in q_lower for k in [
        "name", "holder", "account", "ifsc", "branch", "bank",
        "address", "micr", "period", "who", "customer", "balance"
    ])
    if is_account_query:
        if "p_0" in parent_map and "p_0" not in parent_ids:
            parent_ids.insert(0, "p_0")
        parent_ids = parent_ids[:2]
    else:
        parent_ids = parent_ids[:3]

    # 4. Assemble context from in-memory parent_map (instant, 0.00ms, no Chroma crash)
    context_blocks = []
    if is_account_query and first_page_text:
        context_blocks.append(first_page_text)

    for pid in parent_ids:
        if pid in parent_map and parent_map[pid] not in context_blocks:
            context_blocks.append(parent_map[pid])

    # Fallback to hit chunks if no parent blocks were assembled
    if not context_blocks:
        context_blocks = [h.page_content for h in hits if h.page_content]

    if not context_blocks and first_page_text:
        context_blocks = [first_page_text]

    if not context_blocks:
        return "I couldn't find any relevant data in the statement to answer that."

    # Cap context size (~8000 chars) to stay well within free-tier TPM rate limits
    context_text = "\n---\n".join(context_blocks)[:8000]

    # 5. Build and run chain
    template = """You are a Senior Financial Auditor.
Answer the user query accurately and directly using the bank statement data below.
If the information (such as account holder name, account number, or amount) is present in the context, state it clearly.

Context:
{context}

Question: {question}
Answer:"""

    # 5. Execute via Groq (fast real-time chat) with automatic Gemini fallback
    formatted_prompt = template.format(context=context_text, question=user_query)
    if os.getenv("GROQ_API_KEY"):
        try:
            return _groq_chat(formatted_prompt, temperature=0)
        except Exception:
            if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                return _gemini_generate(formatted_prompt, temperature=0)
            raise
    return _gemini_generate(formatted_prompt, temperature=0)



def _batch_with_retry(batch_text, retries=2):
    prompt = (
        "You are a high-precision Bank Audit bot. Your task is to extract EVERY transaction row. "
        "SBI statements often have dates like '01-Jan-2024 (01-Jan-2024)' and no S.No. "
        "DO NOT SKIP THESE. Extract the first date, the description, and the amounts. "
        "If a Debit or Credit is missing/blank, use 0.0.\n\n"
        f"Text:\n{batch_text}"
    )
    for attempt in range(retries):
        try:
            res = _gemini_generate(prompt, schema=TransactionBatch, temperature=0)
            if res and res.transactions:
                return res.transactions
        except Exception as e:
            print(f"❌ Batch Error (attempt {attempt+1}): {e}")
            time.sleep(1)
    return []


def get_header_direct(first_page_text):
    prompt = f"Extract identity details. Use EXACT values. Do not guess.\n\nText: {first_page_text}"
    try:
        return _gemini_generate(prompt, schema=AccountDetails, temperature=0)
    except Exception:
        if os.getenv("GROQ_API_KEY"):
            json_prompt = (
                "Extract account identity details as JSON matching keys: "
                "customer_name, account_number, account_type, ifsc_code, micr_code, "
                "branch_name, customer_address, branch_address, statement_period. "
                "Use 'N/A' if missing.\n\nText: " + first_page_text
            )
            raw_json = _groq_chat(json_prompt, temperature=0, json_mode=True)
            return AccountDetails.model_validate_json(raw_json)
        raise


@st.cache_data(show_spinner=False)
def _cached_transactions(full_text: str):
    """Cache parsed transactions to avoid re-running LLM on same content."""
    batches = [full_text[i:i+25000] for i in range(0, len(full_text), 25000)]

    all_txns = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = executor.map(_batch_with_retry, batches)
    for res in results:
        all_txns.extend(res)
    return all_txns


@dataclass
class PerformanceMetrics:
    latency_sec: float
    throughput_rows_sec: float
    estimated_tokens: int
    cost_usd: float


async def async_process_batch(batch_text, semaphore, retries=2):
    """Processes large statement batches via Gemini with Groq fallback."""
    async with semaphore:
        prompt = (
            "You are a high-precision Bank Auditor. Extract EVERY transaction. "
            "Handle SBI dates like '01-Jan-2024 (01-Jan-2024)'. "
            "Fields: sl_no, txn_date, value_date, description, debit, credit, balance, category. "
            f"\n\nText:\n{batch_text}"
        )
        for attempt in range(retries):
            try:
                res = await asyncio.to_thread(_gemini_generate, prompt, TransactionBatch, 0)
                return res.transactions
            except Exception as e:
                if "429" in str(e).lower() or "quota" in str(e).lower():
                    if attempt < retries - 1:
                        await asyncio.sleep(1.5)
                        continue
                    if os.getenv("GROQ_API_KEY"):
                        try:
                            json_prompt = (
                                "Extract all bank transactions as JSON: {\"transactions\": [{\"sl_no\": int, "
                                "\"txn_date\": str, \"value_date\": str, \"description\": str, \"debit\": float, "
                                "\"credit\": float, \"balance\": float, \"category\": \"Other\"}]}.\n\nText:\n"
                                + batch_text
                            )
                            raw_json = await asyncio.to_thread(_groq_chat, json_prompt, 0, True)
                            return TransactionBatch.model_validate_json(raw_json).transactions
                        except Exception:
                            return []
                return []
    return []


async def run_parallel_extraction(text_batches):
    semaphore = asyncio.Semaphore(3)
    tasks = [async_process_batch(batch, semaphore) for batch in text_batches]
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]


def get_detailed_report(opening_balance, closing_balance, first_page_text, raw_docs):
    start_time = time.perf_counter()

    account_info = get_header_direct(first_page_text)

    full_text = "\n".join([d.page_content for d in raw_docs])

    # Gemini's 1M context handles 25,000-char batches cleanly -> fewer total API calls
    text_batches = [full_text[i:i+25000] for i in range(0, len(full_text), 25000)]

    all_txns = asyncio.run(run_parallel_extraction(text_batches))

    # FIX BUG 8: Use local classify_transaction instead of LLM for categories
    # This is 0.1ms per transaction vs ~200ms per transaction via LLM
    for txn in all_txns:
        if not txn.category or txn.category == "Other":
            classified = classify_transaction(txn.description)
            if classified:
                txn.category = classified.get("category", txn.category)

    duration = time.perf_counter() - start_time

    total_debits = sum(t.debit for t in all_txns)
    total_credits = sum(t.credit for t in all_txns)

    metrics = {
        "latency_sec": duration,
        "throughput_rows_sec": len(all_txns) / duration if duration > 0 else 0,
        "estimated_tokens": len(full_text) // 4,
        "cost_usd": (len(full_text) // 4 / 1_000_000) * 0.20
    }
    st.session_state.last_run_metrics = metrics
    print(f"BENCHMARK: Processed {len(all_txns)} rows in {duration:.2f}s")

    return FullStatementReport(
        account_info=account_info,
        transactions=all_txns,
        total_debits=total_debits,
        total_credits=total_credits,
        opening_balance=opening_balance,
        closing_balance=closing_balance
    )


def get_finance_advice_stream(user_query: str, vectorstore):
    """Yield advice response using RAG (Groq primary, Gemini fallback)."""
    yield get_finance_advice(user_query, vectorstore)


# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Colour Palette ─────────────────────────────────────────────────────────────
_NAVY        = colors.HexColor("#0f172a")   # headings / header bg
_BLUE        = colors.HexColor("#3b82f6")   # accent / borders
_BLUE_LIGHT  = colors.HexColor("#eff6ff")   # metric card bg (blue)
_BLUE_MID    = colors.HexColor("#bfdbfe")   # metric card border
_GREEN       = colors.HexColor("#10b981")   # credit accent
_GREEN_LIGHT = colors.HexColor("#d1fae5")
_GREEN_MID   = colors.HexColor("#6ee7b7")
_RED         = colors.HexColor("#ef4444")
_RED_LIGHT   = colors.HexColor("#fee2e2")
_RED_MID     = colors.HexColor("#fca5a5")
_RED_TXT     = colors.HexColor("#991b1b")
_AMBER_LIGHT = colors.HexColor("#fffbeb")
_AMBER_MID   = colors.HexColor("#fcd34d")
_AMBER_TXT   = colors.HexColor("#92400e")
_SLATE       = colors.HexColor("#1e293b")   # section headers
_MUTED       = colors.HexColor("#64748b")   # labels / captions
_BORDER      = colors.HexColor("#e2e8f0")   # table lines
_ROW_EVEN    = colors.HexColor("#f8fafc")   # zebra rows
_WHITE       = colors.white
_INDIGO_LIGHT = colors.HexColor("#ede9fe")
_INDIGO      = colors.HexColor("#4f46e5")


# ── Helper: quick ParagraphStyle ──────────────────────────────────────────────
_style_cache: dict = {}

def _style(name, **kw):
    if name not in _style_cache:
        _style_cache[name] = ParagraphStyle(name, **kw)
    return _style_cache[name]


# ── Custom Flowable: Solid colour band (used as section dividers/headers) ─────
class ColourBand(Flowable):
    """A thin coloured rectangle, full content-width."""
    def __init__(self, height=2, color=_BLUE, width_frac=1.0):
        super().__init__()
        self._h = height
        self._color = color
        self._frac = width_frac

    def draw(self):
        self.canv.setFillColor(self._color)
        w = (self._width if hasattr(self, "_width") else 150) * self._frac
        self.canv.rect(0, 0, w, self._h, fill=1, stroke=0)

    def wrap(self, avail_w, avail_h):
        self._width = avail_w
        return avail_w * self._frac, self._h


# ── Page template with header & footer drawn on canvas ────────────────────────
def _make_page_template(doc, report_data):
    PAGE_W, PAGE_H = A4
    MARGIN = doc.leftMargin

    def header_footer(canvas, doc_obj):
        canvas.saveState()
        ai = report_data.account_info

        # ── Header bar ──────────────────────────────────────────────────────
        bar_h = 14 * mm
        canvas.setFillColor(_NAVY)
        canvas.rect(0, PAGE_H - bar_h, PAGE_W, bar_h, fill=1, stroke=0)

        # Logo circle
        cx, cy = MARGIN + 5 * mm, PAGE_H - bar_h / 2
        canvas.setFillColor(_BLUE)
        canvas.circle(cx, cy, 4.5 * mm, fill=1, stroke=0)
        canvas.setFillColor(_WHITE)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawCentredString(cx, cy - 3, "AI")

        # App name
        canvas.setFont("Helvetica-Bold", 11)
        canvas.setFillColor(_WHITE)
        canvas.drawString(MARGIN + 12 * mm, PAGE_H - bar_h / 2 - 4, "FinAdvisor  ·  Audit Report")

        # Right: customer + page
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawRightString(PAGE_W - MARGIN,
                               PAGE_H - bar_h / 2 - 4,
                               f"{ai.customer_name}  ·  Page {doc_obj.page}")

        # Thin accent line under header
        canvas.setStrokeColor(_BLUE)
        canvas.setLineWidth(1.5)
        canvas.line(0, PAGE_H - bar_h - 0.8, PAGE_W, PAGE_H - bar_h - 0.8)

        # ── Footer bar ───────────────────────────────────────────────────────
        canvas.setFillColor(_ROW_EVEN)
        canvas.rect(0, 0, PAGE_W, 10 * mm, fill=1, stroke=0)

        canvas.setStrokeColor(_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 10 * mm, PAGE_W - MARGIN, 10 * mm)

        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(_MUTED)
        now = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
        canvas.drawString(MARGIN, 3.5 * mm,
                          f"Generated by AI Personal Finance Advisor  ·  {now}  ·  Confidential")
        canvas.drawRightString(PAGE_W - MARGIN, 3.5 * mm,
                               f"A/C: {ai.account_number}")

        canvas.restoreState()

    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="main",
    )
    return PageTemplate(id="main_tmpl", frames=[frame], onPage=header_footer)


# ── Main function ──────────────────────────────────────────────────────────────
def generate_pdf_report(
    report_data,
    plotly_fig,
    anomalies_text: Optional[str] = None,
) -> BytesIO:
    buf = BytesIO()
    PAGE_W, PAGE_H = A4
    MARGIN   = 20 * mm
    T_MARGIN = 22 * mm   # extra top — clears the header bar
    B_MARGIN = 16 * mm   # extra bottom — clears the footer bar
    CW = PAGE_W - 2 * MARGIN

    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=T_MARGIN, bottomMargin=B_MARGIN,
    )
    doc.addPageTemplates([_make_page_template(doc, report_data)])

    # ── Shared styles ──────────────────────────────────────────────────────────
    def S(name, **kw):
        return _style(f"pdf_{name}", **kw)

    TITLE   = S("title",  fontName="Helvetica-Bold", fontSize=22, textColor=_SLATE,
                alignment=TA_CENTER, spaceAfter=4, leading=26)
    SUBTITLE= S("sub",    fontSize=10, textColor=_MUTED,
                alignment=TA_CENTER, spaceAfter=0, leading=14)
    SECTION = S("sec",    fontName="Helvetica-Bold", fontSize=11, textColor=_SLATE,
                spaceBefore=14, spaceAfter=8, leading=14)
    CELL    = S("cell",   fontSize=9,  leading=13, textColor=_SLATE)
    CELL_B  = S("cellb",  fontName="Helvetica-Bold", fontSize=9, leading=13, textColor=_SLATE)
    CELL_M  = S("cellm",  fontSize=9,  leading=13, textColor=_MUTED)
    ALERT   = S("alert",  fontSize=8.5, leading=13, textColor=_RED_TXT)
    FOOT    = S("foot",   fontSize=7.5, textColor=_MUTED, alignment=TA_CENTER)
    MONO    = S("mono",   fontName="Courier", fontSize=9, leading=13, textColor=_SLATE)
    MONO_R  = S("monor",  fontName="Courier", fontSize=9, leading=13,
                alignment=TA_RIGHT, textColor=_SLATE)

    def p(text, style=CELL):
        return Paragraph(text, style)

    def sp(h=6):
        return Spacer(1, h)

    def hr(before=4, after=8, color=_BORDER, thick=0.5):
        return HRFlowable(width="100%", thickness=thick, color=color,
                          spaceBefore=before, spaceAfter=after)

    story = []

    # ══════════════════════════════════════════════════════════════════════════
    # 1. TITLE BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    ai = report_data.account_info
    story.append(sp(6))
    story.append(p("AI Financial Audit Report", TITLE))
    story.append(p(f"Statement Period: {ai.statement_period}", SUBTITLE))
    story.append(sp(10))
    story.append(ColourBand(height=3, color=_BLUE))
    story.append(sp(16))

    # ══════════════════════════════════════════════════════════════════════════
    # 2. ACCOUNT INFORMATION
    # ══════════════════════════════════════════════════════════════════════════
    story.append(p("Account Information", SECTION))

    HALF = CW / 2
    acc_rows = [
        # header row
        [p("<b>Field</b>", CELL_B), p("<b>Details</b>", CELL_B),
         p("<b>Field</b>", CELL_B), p("<b>Details</b>", CELL_B)],
        [p("Customer Name", CELL_M), p(ai.customer_name, CELL_B),
         p("IFSC Code",      CELL_M), p(ai.ifsc_code, MONO)],
        [p("Account Number", CELL_M), p(ai.account_number, MONO),
         p("Branch",         CELL_M), p(ai.branch_name, CELL)],
        [p("Account Type",   CELL_M), p(ai.account_type, CELL),
         p("Period",         CELL_M), p(ai.statement_period, CELL)],
    ]
    COL4 = [CW * 0.18, CW * 0.32, CW * 0.18, CW * 0.32]
    acc_tbl = Table(acc_rows, colWidths=COL4, hAlign="LEFT")
    acc_tbl.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",    (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), _WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
        # Zebra
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_WHITE, _ROW_EVEN]),
        # Box
        ("BOX",           (0, 0), (-1, -1), 0.8, _BLUE),
        ("INNERGRID",     (0, 0), (-1, -1), 0.4, _BORDER),
        # Padding
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(acc_tbl)
    story.append(sp(16))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. KEY METRICS  (4 cards in a row)
    # ══════════════════════════════════════════════════════════════════════════
    story.append(hr(before=2, after=10))
    story.append(p("Key Metrics", SECTION))

    metrics = [
        ("Opening Balance", f"Rs.{report_data.opening_balance:,.2f}", _BLUE_LIGHT,  _BLUE,  _BLUE_MID),
        ("Total Debits",    f"Rs.{report_data.total_debits:,.2f}",    _RED_LIGHT,   _RED,   _RED_MID),
        ("Total Credits",   f"Rs.{report_data.total_credits:,.2f}",   _GREEN_LIGHT, _GREEN, _GREEN_MID),
        ("Closing Balance", f"Rs.{report_data.closing_balance:,.2f}", _BLUE_LIGHT,  _BLUE,  _BLUE_MID),
    ]

    def _metric_cell(label, value, bg, fg, border_c):
        val_style = _style(f"mv_{label.replace(' ','_')}",
                           fontName="Helvetica-Bold", fontSize=13,
                           textColor=fg, alignment=TA_CENTER, leading=16)
        lbl_style = _style(f"ml_{label.replace(' ','_')}",
                           fontSize=8, textColor=_MUTED,
                           alignment=TA_CENTER, leading=11)
        return [Paragraph(value, val_style), Paragraph(label, lbl_style)]

    mcells  = [[_metric_cell(*m) for m in metrics]]
    CW4     = CW / 4
    met_tbl = Table(mcells, colWidths=[CW4] * 4, rowHeights=[54], hAlign="LEFT")
    bg_cmds = [("BACKGROUND", (i, 0), (i, 0), metrics[i][2]) for i in range(4)]
    bx_cmds = [("BOX",        (i, 0), (i, 0), 1.0, metrics[i][3]) for i in range(4)]
    met_tbl.setStyle(TableStyle(bg_cmds + bx_cmds + [
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    story.append(met_tbl)
    story.append(sp(16))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. TRANSACTION SUMMARY TABLE  (top 10 debits)
    # ══════════════════════════════════════════════════════════════════════════
    story.append(hr(before=2, after=10))
    story.append(p("Transaction Summary", SECTION))

    txns = getattr(report_data, "transactions", [])
    debits = sorted(
        [t for t in txns if getattr(t, "debit", 0) > 0],
        key=lambda t: t.debit, reverse=True
    )[:12]

    if debits:
        txn_header = [
            p("<b>Date</b>",        CELL_B),
            p("<b>Description</b>", CELL_B),
            p("<b>Category</b>",    CELL_B),
            p("<b>Amount (Rs.)</b>",CELL_B),
        ]
        txn_rows = [txn_header]
        for t in debits:
            txn_rows.append([
                p(str(getattr(t, "txn_date", "—")), CELL_M),
                p(str(getattr(t, "description", "—"))[:50], CELL),
                p(str(getattr(t, "category",    "—")), CELL),
                p(f'{getattr(t, "debit", 0):,.2f}', MONO_R),
            ])

        TXN_COLS = [CW * 0.13, CW * 0.45, CW * 0.22, CW * 0.20]
        txn_tbl  = Table(txn_rows, colWidths=TXN_COLS, hAlign="LEFT", repeatRows=1)
        txn_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), _NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0), _WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_WHITE, _ROW_EVEN]),
            ("BOX",           (0, 0), (-1, -1), 0.8, _BLUE),
            ("INNERGRID",     (0, 0), (-1, -1), 0.3, _BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",         (3, 0), (3, -1),  "RIGHT"),
        ]))
        story.append(txn_tbl)
        story.append(sp(16))

    # ══════════════════════════════════════════════════════════════════════════
    # 5. SPENDING BREAKDOWN CHART
    # ══════════════════════════════════════════════════════════════════════════
    if plotly_fig is not None:
        story.append(hr(before=2, after=10))
        story.append(p("Spending Breakdown", SECTION))

        fig_copy = plotly_fig
        fig_copy.update_layout(
            margin=dict(l=20, r=20, t=20, b=90),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Helvetica", color="#1e293b", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=-0.42,
                        font=dict(size=9)),
        )
        try:
            img_bytes = pio.to_image(fig_copy, format="png", width=800, height=380, scale=2)
            chart_h   = CW * 0.48
            img = Image(BytesIO(img_bytes), width=CW, height=chart_h)
            # Wrap chart in a light card (drawn via Table with bg)
            chart_card = Table([[img]], colWidths=[CW])
            chart_card.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), _ROW_EVEN),
                ("BOX",           (0, 0), (-1, -1), 0.8, _BORDER),
                ("TOPPADDING",    (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0,  0), (-1, -1), 8),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ]))
            story.append(chart_card)
            story.append(sp(16))
        except Exception as e:
            # Kaleido/Chrome not available on deployed platforms - skip chart in PDF
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # 6. AI SECURITY ALERTS
    # ══════════════════════════════════════════════════════════════════════════
    if anomalies_text:
        story.append(hr(before=2, after=10))
        story.append(p("AI Security Alerts", SECTION))

        lines = [
            ln.strip().lstrip("•·-–* ").strip()
            for ln in anomalies_text.splitlines()
            if ln.strip()
        ]

        ICON_W = 12
        TEXT_W = CW - ICON_W

        alert_rows = []
        for line in lines:
            if not line:
                continue
            icon_style = _style("alert_icon",
                                fontName="Helvetica-Bold", fontSize=10,
                                textColor=_RED, alignment=TA_CENTER, leading=14)
            txt_style  = _style("alert_txt",
                                fontSize=8.5, leading=13.5, textColor=_RED_TXT)
            alert_rows.append([
                Paragraph("!", icon_style),
                Paragraph(line, txt_style),
            ])

        if alert_rows:
            a_tbl = Table(alert_rows, colWidths=[ICON_W, TEXT_W], hAlign="LEFT")
            a_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), _AMBER_LIGHT),
                ("BOX",           (0, 0), (-1, -1), 0.8, _AMBER_MID),
                ("INNERGRID",     (0, 0), (-1, -1), 0.3, colors.HexColor("#fde68a")),
                ("TOPPADDING",    (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING",   (0, 0), (0, -1),  4),
                ("RIGHTPADDING",  (0, 0), (0, -1),  4),
                ("LEFTPADDING",   (1, 0), (1, -1),  10),
                ("RIGHTPADDING",  (1, 0), (1, -1),  10),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("ALIGN",         (0, 0), (0, -1),  "CENTER"),
                ("TEXTCOLOR",     (0, 0), (0, -1),  _AMBER_TXT),
            ]))
            story.append(KeepTogether(a_tbl))
        else:
            story.append(p(anomalies_text, CELL))

        story.append(sp(12))

    # ── Build ────────────────────────────────────────────────────────────────
    doc.build(story)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# Forecast
# ══════════════════════════════════════════════════════════════════════════════

def calculate_forecast(transactions):
    df = pd.DataFrame([t.model_dump() for t in transactions])
    # df["date_dt"] = pd.to_datetime(df["txn_date"], infer_datetime_format=True, errors="coerce")
    df["date_dt"] = pd.to_datetime(df["txn_date"], format="%d-%m-%Y", errors="coerce")
    monthly = (
        df[df["debit"] > 0]
        .groupby(df["date_dt"].dt.to_period("M"))["debit"]
        .sum()
        .reset_index()
    )
    monthly["date_dt"] = monthly["date_dt"].dt.to_timestamp()

    if len(monthly) < 2:
        return None, "Insufficient data (need 2+ months)"

    avg_growth = monthly["debit"].pct_change().mean()
    prediction = monthly["debit"].iloc[-1] * (1 + (avg_growth or 0))
    next_date  = monthly["date_dt"].iloc[-1] + pd.DateOffset(months=1)
    return (monthly, next_date, prediction), None


def get_forecast_insights(monthly_data, prediction):
    prompt = (
        f"Analyze these monthly totals:\n{monthly_data.to_string()}\n"
        f"Prediction: ₹{prediction:,.2f}. Provide reasoning."
    )
    try:
        return _gemini_generate(prompt, schema=ForecastInsight, temperature=0)
    except Exception:
        if os.getenv("GROQ_API_KEY"):
            json_prompt = (
                f"{prompt}\n\nReturn JSON with keys: "
                "trend_analysis (str), risk_warnings (list of str), saving_tips (list of 3 str)."
            )
            raw_json = _groq_chat(json_prompt, temperature=0, json_mode=True)
            return ForecastInsight.model_validate_json(raw_json)
        raise


def fast_intent(text: str):
    t = text.lower()
    if "report" in t or "audit" in t:       return "audit"
    if "budget" in t:                        return "budget"
    if "forecast" in t or "future" in t:    return "forecast"
    return "chat"