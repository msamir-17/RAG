import time
import pandas as pd
import plotly.io as pio
from io import BytesIO
from typing import List, Optional
from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, Image, HRFlowable, KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import streamlit as st
from modules.schema import FullStatementReport, AccountDetails, Transaction, ForecastInsight


class TransactionBatch(BaseModel):
    transactions: List[Transaction]


# ── LLM helpers ───────────────────────────────────────────────────────────────

def get_finance_advice(user_query: str, vectorstore) -> str:
    llm       = ChatMistralAI(model="mistral-small-2506")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    template  = """You are a concise Financial Advisor. Answer based ONLY on context.
Context: {context}
Question: {question}
Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain  = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain.invoke(user_query)


def _batch_with_retry(batch_text, batch_llm, retries=2):
    for _ in range(retries):
        try:
            return batch_llm.invoke(batch_text).transactions
        except Exception:
            time.sleep(1)
    return []


def get_header_direct(first_page_text):
    llm            = ChatMistralAI(model="mistral-small-2506", temperature=0)
    structured_llm = llm.with_structured_output(AccountDetails)
    prompt = f"Extract identity details. Use EXACT values. Do not guess.\n\nText: {first_page_text}"
    return structured_llm.invoke(prompt)


# def get_detailed_report(opening_balance, closing_balance, first_page_text, raw_docs):
#     account_info = get_header_direct(first_page_text)
#     llm          = ChatMistralAI(model="mistral-small-2506", temperature=0)
#     batch_llm    = llm.with_structured_output(TransactionBatch)
#     full_text    = "\n".join([d.page_content for d in raw_docs])
#     batches      = [full_text[i:i+6000] for i in range(0, len(full_text), 6000)]

#     all_txns = []
#     for batch in batches:
#         all_txns.extend(_batch_with_retry(batch, batch_llm))

#     return FullStatementReport(
#         account_info    = account_info,
#         transactions    = all_txns,
#         total_debits    = sum(t.debit  for t in all_txns),
#         total_credits   = sum(t.credit for t in all_txns),
#         opening_balance = opening_balance,
#         closing_balance = closing_balance,
#     )

from concurrent.futures import ThreadPoolExecutor

@st.cache_data(show_spinner=False)
def _cached_transactions(full_text: str):
    """Cache parsed transactions to avoid re-running LLM"""
    
    llm = ChatMistralAI(model="mistral-small-2506", temperature=0)
    batch_llm = llm.with_structured_output(TransactionBatch)

    batches = [full_text[i:i+6000] for i in range(0, len(full_text), 6000)]

    def process_batch(batch):
        return _batch_with_retry(batch, batch_llm)

    all_txns = []

    # ⚡ Parallel execution (FASTER)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_batch, batches)

    for res in results:
        all_txns.extend(res)

    return all_txns


def get_detailed_report(opening_balance, closing_balance, first_page_text, raw_docs):
    # ✅ Keep this (IMPORTANT)
    account_info = get_header_direct(first_page_text)

    # ✅ Prepare text
    full_text = "\n".join([d.page_content for d in raw_docs])

    # ✅ Use cached + parallel version
    all_txns = _cached_transactions(full_text)

    # ✅ Final report (same as before)
    return FullStatementReport(
        account_info    = account_info,
        transactions    = all_txns,
        total_debits    = sum(t.debit for t in all_txns),
        total_credits   = sum(t.credit for t in all_txns),
        opening_balance = opening_balance,
        closing_balance = closing_balance,
    )



# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# Colour palette
_INDIGO  = colors.HexColor("#4f46e5")
_SLATE   = colors.HexColor("#1e293b")
_MUTED   = colors.HexColor("#64748b")
_INDIGO_LIGHT = colors.HexColor("#ede9fe")
_INDIGO_MID   = colors.HexColor("#c7d2fe")
_RED_BG  = colors.HexColor("#fff1f2")
_RED_BRD = colors.HexColor("#fecdd3")
_RED_TXT = colors.HexColor("#9f1239")
_GREEN   = colors.HexColor("#059669")
_BORDER  = colors.HexColor("#e2e8f0")


def _style(name, **kw) -> ParagraphStyle:
    base = {"fontName": "Helvetica", "fontSize": 10, "leading": 14,
            "textColor": _SLATE, "spaceAfter": 0, "spaceBefore": 0}
    base.update(kw)
    return ParagraphStyle(name, **base)


def generate_pdf_report(
    report_data: FullStatementReport,
    plotly_fig,
    anomalies_text: Optional[str] = None,
) -> BytesIO:
    """
    Full-page PDF:
      1. Title
      2. Account Information
      3. Key Metrics
      4. Spending Breakdown chart
      5. AI Security Alerts
    """
    buf = BytesIO()
    PAGE_W, PAGE_H = A4
    MARGIN = 18 * mm
    CW = PAGE_W - 2 * MARGIN          # content width

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )

    # ── Styles ────────────────────────────────────────────────────────────────
    S = {
        "title":      _style("s_title",    fontName="Helvetica-Bold", fontSize=20,
                             textColor=_INDIGO, alignment=TA_CENTER, spaceAfter=2),
        "subtitle":   _style("s_sub",      fontSize=9, textColor=_MUTED,
                             alignment=TA_CENTER, spaceAfter=14),
        "section":    _style("s_sec",      fontName="Helvetica-Bold", fontSize=12,
                             textColor=_SLATE, spaceBefore=12, spaceAfter=6),
        "cell":       _style("s_cell",     fontSize=9,  leading=13),
        "cell_bold":  _style("s_cellb",    fontName="Helvetica-Bold", fontSize=9, leading=13),
        "metric_val": _style("s_mval",     fontName="Helvetica-Bold", fontSize=14,
                             alignment=TA_CENTER),
        "metric_lbl": _style("s_mlbl",     fontSize=8, textColor=_MUTED,
                             alignment=TA_CENTER),
        "alert":      _style("s_alert",    fontSize=8.5, leading=13, textColor=_RED_TXT),
        "footer":     _style("s_foot",     fontSize=7.5, textColor=_MUTED,
                             alignment=TA_CENTER),
    }

    def hr(space_before=4, space_after=10, color=_BORDER, thick=0.6):
        return HRFlowable(width="100%", thickness=thick, color=color,
                          spaceBefore=space_before, spaceAfter=space_after)

    def p(text, style="cell"):
        return Paragraph(text, S[style])

    story = []

    # ── 1. Title ──────────────────────────────────────────────────────────────
    story.append(p("AI Financial Audit Report", "title"))
    story.append(p(f"Statement Period: {report_data.account_info.statement_period}", "subtitle"))
    story.append(hr(space_before=0, space_after=14, color=_INDIGO, thick=1.5))

    # ── 2. Account Information ────────────────────────────────────────────────
    story.append(p("🏦  Account Information", "section"))

    ai = report_data.account_info
    rows = [
        [p(f"<b>Customer:</b>  {ai.customer_name}"),   p(f"<b>IFSC:</b>  {ai.ifsc_code}")],
        [p(f"<b>A/C Number:</b>  {ai.account_number}"), p(f"<b>Branch:</b>  {ai.branch_name}")],
        [p(f"<b>Account Type:</b>  {ai.account_type}"), p(f"<b>Period:</b>  {ai.statement_period}")],
    ]
    acc_tbl = Table(rows, colWidths=[CW * 0.5, CW * 0.5], hAlign="LEFT")
    acc_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _INDIGO_LIGHT),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_INDIGO_LIGHT, colors.HexColor("#f5f3ff")]),
        ("BOX",       (0, 0), (-1, -1), 0.7, _INDIGO),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, _INDIGO_MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(acc_tbl)
    story.append(Spacer(1, 12))

    # ── 3. Key Metrics ────────────────────────────────────────────────────────
    story.append(p("📊  Key Metrics", "section"))

    metrics = [
        ("Opening Balance", f"₹{report_data.opening_balance:,.2f}", "#ede9fe", "#4f46e5"),
        ("Total Debits",    f"₹{report_data.total_debits:,.2f}",    "#fee2e2", "#dc2626"),
        ("Total Credits",   f"₹{report_data.total_credits:,.2f}",   "#d1fae5", "#059669"),
        ("Closing Balance", f"₹{report_data.closing_balance:,.2f}", "#ede9fe", "#4f46e5"),
    ]

    def metric_cell(label, value, bg, fg):
        return [
            Paragraph(
                f'<font color="{fg}"><b>{value}</b></font>',
                ParagraphStyle(f"mv_{label}", fontName="Helvetica-Bold", fontSize=13,
                               textColor=colors.HexColor(fg), alignment=TA_CENTER,
                               leading=16),
            ),
            Paragraph(
                label,
                ParagraphStyle(f"ml_{label}", fontSize=8, textColor=_MUTED,
                               alignment=TA_CENTER, leading=11),
            ),
        ]

    mcells  = [[metric_cell(*m) for m in metrics]]
    met_tbl = Table(mcells, colWidths=[CW / 4] * 4, rowHeights=[52], hAlign="LEFT")
    met_tbl.setStyle(TableStyle(
        [("BACKGROUND",   (i, 0), (i, 0), colors.HexColor(metrics[i][2])) for i in range(4)] +
        [("BOX",          (i, 0), (i, 0), 0.6, colors.HexColor(metrics[i][3])) for i in range(4)] +
        [
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ]
    ))
    story.append(met_tbl)
    story.append(Spacer(1, 14))

    # ── 4. Spending Breakdown chart ───────────────────────────────────────────
    if plotly_fig is not None:
        story.append(hr())
        story.append(p("📈  Spending Breakdown", "section"))

        fig_copy = plotly_fig
        fig_copy.update_layout(
            margin=dict(l=10, r=10, t=10, b=80),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#1e293b", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        )
        img_bytes = pio.to_image(fig_copy, format="png", width=780, height=370, scale=2)
        chart_h   = CW * 0.47        # proportional height
        story.append(Image(BytesIO(img_bytes), width=CW, height=chart_h))
        story.append(Spacer(1, 12))

    # ── 5. AI Security Alerts ─────────────────────────────────────────────────
    if anomalies_text:
        story.append(hr())
        story.append(p("🚩  AI Security Alerts", "section"))

        lines = [ln.strip().lstrip("•·-–* ").strip()
                 for ln in anomalies_text.splitlines() if ln.strip()]

        alert_rows = []
        for line in lines:
            if not line:
                continue
            alert_rows.append([
                Paragraph("▸", ParagraphStyle(
                    "icon", fontName="Helvetica-Bold", fontSize=9,
                    textColor=_RED_TXT, alignment=TA_CENTER, leading=13,
                )),
                Paragraph(line, ParagraphStyle(
                    "aline", fontSize=8.5, leading=13, textColor=_RED_TXT,
                    leftIndent=0,
                )),
            ])

        if alert_rows:
            ICON_W = 14  # wide enough to hold padding without going negative
            a_tbl = Table(alert_rows, colWidths=[ICON_W, CW - ICON_W], hAlign="LEFT")
            a_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), _RED_BG),
                ("BOX",           (0, 0), (-1, -1), 0.8, _RED_BRD),
                ("INNERGRID",     (0, 0), (-1, -1), 0.3, _RED_BRD),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING",   (0, 0), (0, -1),  2),   # icon col: minimal padding
                ("RIGHTPADDING",  (0, 0), (0, -1),  2),
                ("LEFTPADDING",   (1, 0), (1, -1),  8),   # text col: normal padding
                ("RIGHTPADDING",  (1, 0), (1, -1),  8),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("ALIGN",         (0, 0), (0, -1),  "CENTER"),
            ]))
            story.append(a_tbl)   # removed KeepTogether — it was the crash wrapper
        else:
            story.append(p(anomalies_text))

        story.append(Spacer(1, 10))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(hr(space_before=8, space_after=5))
    story.append(p("Generated by AI Personal Finance Advisor  ·  Confidential", "footer"))

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
    llm            = ChatMistralAI(model="mistral-small-2506", temperature=0)
    structured_llm = llm.with_structured_output(ForecastInsight)
    prompt = (
        f"Analyze these monthly totals:\n{monthly_data.to_string()}\n"
        f"Prediction: ₹{prediction:,.2f}. Provide reasoning."
    )
    return structured_llm.invoke(prompt)


def fast_intent(text: str):
    t = text.lower()
    if "report" in t or "audit" in t:       return "audit"
    if "budget" in t:                        return "budget"
    if "forecast" in t or "future" in t:    return "forecast"
    return "chat"