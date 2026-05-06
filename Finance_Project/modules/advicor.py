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



from reportlab.platypus.flowables import Flowable
import plotly.io as pio




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
        img_bytes = pio.to_image(fig_copy, format="png", width=800, height=380, scale=2)
        chart_h   = CW * 0.48
        img = Image(BytesIO(img_bytes), width=CW, height=chart_h)
        # Wrap chart in a light card (drawn via Table with bg)
        chart_card = Table([[img]], colWidths=[CW])
        chart_card.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), _ROW_EVEN),
            ("BOX",           (0, 0), (-1, -1), 0.8, _BORDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ]))
        story.append(chart_card)
        story.append(sp(16))

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