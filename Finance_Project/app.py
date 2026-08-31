import html
import os
import re
from eval.snapshots import capture_eval_snapshot
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.advicor import (calculate_forecast,
                              get_detailed_report, get_finance_advice,
                              get_forecast_insights, generate_pdf_report)
from modules.processor import process_pdf_to_memory
from modules.voice import (classify_intent, get_audio_hash, normalize_transcript, transcribe_audio)

def strip_html_tags(text: str) -> str:
    """Remove raw HTML tags from text (e.g. </div>) while keeping the rest."""
    return re.sub(r'<[^>]+>', '', text)


# ── Session state (must be before inject_styles so sidebar_collapsed is initialized) ──
_defaults = {
    "ready": False, "messages": [], "opening_balance": 0.0, "closing_balance": 0.0,
    "active_tab": 0, "voice_nav": None, "last_voice_hash": "",
    "voice_status": "idle", "voice_label": "",
    "theme": "dark", "sidebar_collapsed": False,
    "budgets": {"Food & Dining": 5000, "Travel & Transport": 3000,
                "Shopping": 4000, "Utilities & Bills": 2000},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def inject_styles():
    try:
        css_path = os.path.join(os.path.dirname(__file__), "assets", "theme.css")
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
    except Exception:
        css = ""
    theme = st.session_state.get("theme", "dark")
    # Append light-mode override directly into CSS so it always applies reliably
    if theme == "light":
        css += """
        :root {
            --bg: #f0f2f8 !important;
            --bg-secondary: #e8eaf0 !important;
            --surface: #ffffff !important;
            --surface-hover: #f8f9fc !important;
            --surface-elevated: #ffffff !important;
            --surface-2: #f8f9fc !important;
            --surface-3: #f1f3f9 !important;
            --border: rgba(0,0,0,0.07) !important;
            --border-strong: rgba(0,0,0,0.13) !important;
            --text-primary: #111827 !important;
            --text-secondary: #4b5563 !important;
            --text-muted: #9ca3af !important;
            --shadow-xs: 0 1px 2px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.04) !important;
            --shadow-sm: 0 2px 8px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04) !important;
            --shadow-md: 0 4px 20px rgba(0,0,0,0.08), 0 2px 8px rgba(0,0,0,0.04) !important;
            --shadow-lg: 0 8px 40px rgba(0,0,0,0.10), 0 4px 16px rgba(0,0,0,0.06) !important;
            --shadow-accent: 0 4px 20px rgba(99,102,241,0.15) !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: rgba(255,255,255,0.45) !important;
        }
        [data-testid="stSidebar"] * {
            color: #8892a4 !important;
        }
        [data-testid="stSidebar"] .stRadio label:has(input:checked) {
            color: #c7d2fe !important;
        }
        """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)



# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Finance Advisor", page_icon="💰", layout="wide", initial_sidebar_state="expanded")
inject_styles()
# ── CSS — Premium Fintech Dashboard (Stripe/Razorpay/CRED style) ──────────────


# ── Theme helpers ─────────────────────────────────────────────────────────────
def set_theme(theme: str):
    st.session_state.theme = theme
    st.rerun()

_PLOTLY_DARK = {
    "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#f0f2f8", "family": "Inter, sans-serif"},
    "xaxis": {"gridcolor": "rgba(255,255,255,0.06)", "linecolor": "rgba(255,255,255,0.1)"},
    "yaxis": {"gridcolor": "rgba(255,255,255,0.06)", "linecolor": "rgba(255,255,255,0.1)"},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#9ca3af"}},
}
_PLOTLY_LIGHT = {
    "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#111827", "family": "Inter, sans-serif"},
    "xaxis": {"gridcolor": "rgba(0,0,0,0.06)", "linecolor": "rgba(0,0,0,0.1)"},
    "yaxis": {"gridcolor": "rgba(0,0,0,0.06)", "linecolor": "rgba(0,0,0,0.1)"},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#4b5563"}},
}
_COLOR_PALETTE = ["#6366f1", "#3b82f6", "#10b981", "#f59e0b", "#f43f5e", "#a855f7", "#06b6d4", "#ec4899"]

def get_plotly_theme():
    return _PLOTLY_LIGHT if st.session_state.get("theme", "dark") == "light" else _PLOTLY_DARK

def update_plotly_layout(fig):
    theme = get_plotly_theme()
    fig.update_layout(**theme)
    return fig

# ── UI Helpers ────────────────────────────────────────────────────────────────
def render_metric_card(icon, icon_color, label, value, delta=None, delta_type="positive"):
    delta_html = ""
    if delta:
        cls = "positive" if delta_type == "positive" else "negative"
        delta_html = f'<div class="metric-info delta {cls}">{delta}</div>'
    st.markdown(f"""
    <div class="metric-card-custom">
        <div class="metric-icon-bg {icon_color}">{icon}</div>
        <div class="metric-info">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {delta_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_ai_insight(text):
    st.markdown(f"""
    <div class="ai-insight-card">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <div class="ai-insight-icon">AI</div>
            <div style="font-size:15px;font-weight:700;color:var(--text-primary);">AI Insight</div>
        </div>
        <div class="ai-insight-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)

def txn_icon_emoji(category):
    cmap = {
        "Food & Dining": "🍔", "Groceries": "🛒", "Shopping": "🛍️",
        "Travel & Transport": "🚗", "Entertainment": "🎬",
        "Utilities & Bills": "⚡", "Healthcare": "🏥", "Education": "📚",
        "Investments": "📈", "Insurance": "🛡️", "Loan/EMI": "🏦",
        "Housing & Rent": "🏠", "Bank Transfer": "🏦", "UPI Transfer": "📲",
        "Cash & ATM": "💵", "Income": "💰", "Taxes & Government": "🏛️",
        "Charity & Donations": "🤝", "Other": "📋",
    }
    return cmap.get(category, "💳")

def txn_icon_color(category):
    cmap = {
        "Food & Dining": "amber", "Groceries": "green", "Shopping": "purple",
        "Travel & Transport": "blue", "Entertainment": "purple",
        "Utilities & Bills": "green", "Healthcare": "red", "Education": "blue",
        "Investments": "green", "Insurance": "blue", "Loan/EMI": "red",
        "Housing & Rent": "blue", "Bank Transfer": "purple", "UPI Transfer": "blue",
        "Cash & ATM": "amber", "Income": "green", "Taxes & Government": "red",
        "Charity & Donations": "purple", "Other": "purple",
    }
    return cmap.get(category, "purple")

# ── Constants ─────────────────────────────────────────────────────────────────
TABS = ["💬 Chat Advisor", "📊 Full Audit Report", "🎯 Budget Planner", "🔮 Spending Forecast"]
CORE_CATS = ["Food & Dining", "Travel & Transport", "Shopping", "Utilities & Bills"]


def route_intent(text: str) -> int:
    t = text.lower()
    if any(w in t for w in ["audit","full report","generate report","statement analysis","transaction"]):
        return 1
    if any(w in t for w in ["budget","goal","planner","spending limit"]):
        return 2
    if any(w in t for w in ["forecast","predict","next month","future","trend","projection"]):
        return 3
    return 0


def parse_dates_flexible(series: pd.Series) -> pd.Series:
    """Try multiple date formats so budget data always loads correctly."""
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d-%b-%Y",
                "%m-%d-%Y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(series, format=fmt, errors="raise")
        except Exception:
            pass
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")


# Apply voice navigation before sidebar renders
if st.session_state.get("voice_nav") is not None:
    st.session_state.active_tab = st.session_state.voice_nav
    st.session_state.voice_nav  = None

# ── Header ────────────────────────────────────────────────────────────────────
with st.container():
    if st.session_state.get("sidebar_collapsed"):
        expand_col, header_col1, header_col2 = st.columns([0.4, 3, 1])
        with expand_col:
            if st.button(">>", key="expand_sidebar", help="Expand sidebar"):
                st.session_state.sidebar_collapsed = False
                st.rerun()
    else:
        header_col1, header_col2 = st.columns([3, 1])

    with header_col1:
        st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 28px; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 4px;">
                👋 Welcome back
            </h1>
            <p style="font-size: 14px; color: var(--text-muted); margin: 0;">
                Upload your bank statement · ask questions · track spending · forecast the future
            </p>
        </div>
        """, unsafe_allow_html=True)
    with header_col2:
        st.markdown("""
        <div style="display:flex;justify-content:flex-end;align-items:center;gap:12px;margin-top:8px;">
            <div style="text-align:right;">
                <div style="font-size:13px;font-weight:600;color:var(--text-primary);">John Doe</div>
                <div style="font-size:11px;color:var(--text-muted);">Personal Account</div>
            </div>
            <div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#a855f7);display:flex;align-items:center;justify-content:center;font-size:16px;color:white;font-weight:700;">JD</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
voice_file = None
with st.sidebar:
    # Branding header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
        <div style="width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#6366f1,#a855f7);display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 4px 12px rgba(99,102,241,0.25);">💰</div>
        <div>
            <div style="font-size:16px;font-weight:800;color:#f0f2f8;letter-spacing:-0.02em;">FinSmart AI</div>
            <div style="font-size:11px;color:#8892a4;margin-top:2px;">Personal Finance Advisor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle
    theme_col1, theme_col2 = st.columns([1, 1])
    with theme_col1:
        st.markdown('<div style="font-size:12px;color:#8892a4;padding-top:6px;">🌗 Theme</div>', unsafe_allow_html=True)
    with theme_col2:
        is_light = st.toggle("", value=st.session_state.theme == "light", key="theme_toggle", label_visibility="collapsed")
        new_theme = "light" if is_light else "dark"
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

    st.markdown("<div style='margin: 12px 0;'></div>", unsafe_allow_html=True)
    st.header("📂  Upload Center")
    uploaded_file = st.file_uploader("Bank statement (PDF)", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_uploaded_file") != file_id:
            for key in list(st.session_state.keys()):
                if key not in ("budgets", "active_tab"):
                    del st.session_state[key]
            st.session_state.update({
                "ready": False, "messages": [], "last_uploaded_file": file_id,
                "last_voice_hash": "", "voice_nav": None,
                "voice_status": "idle", "voice_label": "",
            })
            os.makedirs("data", exist_ok=True)
            path = os.path.join("data", "temp_statement.pdf")
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Reading statement…"):
                result = process_pdf_to_memory(path)
            
            # Unpack result properly
            db, opening, closing, first_page, raw_docs = result
            
            # CHECK FOR ERRORS (db is None means error occurred)
            if db is None:
                error_msg = raw_docs if isinstance(raw_docs, str) else "Unknown error"
                
                # Show appropriate error
                if "INVALID_PDF" in error_msg:
                    st.error("❌ Invalid PDF: No readable text found. Please upload a proper bank statement.")
                elif "ENCRYPTED" in error_msg or "not been decrypted" in error_msg:
                    st.error("❌ Encrypted PDF: Cannot read this file. Try a different PDF.")
                elif "ERROR:" in error_msg:
                    st.error(f"❌ Error: {error_msg}")
                else:
                    st.error(f"❌ Error reading PDF: {error_msg}")
                
                st.session_state.ready = False
                st.session_state.db = None
                st.stop()
            
            # PDF is VALID - Update session state
            st.session_state.update({
                "db": db,
                "opening_balance": opening,
                "closing_balance": closing,
                "first_page_text": first_page,
                "raw_docs": raw_docs,
                "ready": True,
            })
            st.success("✅ Statement loaded!")

    if st.session_state.ready:
        st.divider()

        # Voice section
        st.header("🎤  Voice Command")
        st.markdown(
            '<div class="voice-hint">'
            'Navigate by voice:<br>'
            '<b>💬</b> "chat advisor" &nbsp;·&nbsp; <b>📊</b> "audit report"<br>'
            '<b>🎯</b> "open budget" &nbsp;·&nbsp; <b>🔮</b> "show forecast"<br>'
            'Or ask any financial question!'
            '</div>',
            unsafe_allow_html=True,
        )
        voice_file = st.audio_input("Speak", label_visibility="collapsed")

        # Status chip (clean, no Streamlit error widget)
        vs = st.session_state.get("voice_status", "idle")
        vl = st.session_state.get("voice_label", "")
        if vs == "busy":
            chip = '<div class="voice-chip busy">🎙️ &nbsp;Transcribing…</div>'
        elif vs == "ok" and vl:
            chip = f'<div class="voice-chip ok">✓ &nbsp;{vl}</div>'
        else:
            chip = '<div class="voice-chip idle">🔇 &nbsp;Tap mic to speak</div>'
        st.markdown(chip, unsafe_allow_html=True)

        st.divider()
        st.header("🗂️  Navigate")
        selection = st.radio("Page", TABS, index=st.session_state.active_tab,
                             label_visibility="collapsed")
        clicked = TABS.index(selection)
        if clicked != st.session_state.active_tab:
            st.session_state.active_tab = clicked
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# VOICE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.ready and voice_file:
    audio_bytes  = voice_file.getvalue()
    current_hash = get_audio_hash(audio_bytes)

    if current_hash != st.session_state.get("last_voice_hash", ""):
        st.session_state.last_voice_hash = current_hash
        st.session_state.voice_status   = "busy"
        st.session_state.voice_label    = ""

        with st.spinner("🎙️ Listening…"):
            transcript = transcribe_audio(audio_bytes)

        if transcript:
            transcript    = normalize_transcript(transcript)
            tab_idx       = route_intent(transcript)
            nav_keywords  = {"chat","advisor","back","home","audit","report",
                             "budget","forecast","predict","trend","planner"}
            is_nav        = any(w in transcript.lower() for w in nav_keywords)

            st.session_state.voice_nav    = tab_idx
            st.session_state.voice_status = "ok"
            st.session_state.voice_label  = TABS[tab_idx].split(" ", 1)[1]
            st.toast(f"🚀 {TABS[tab_idx]}", icon="🎙️")

            if tab_idx == 0 and not is_nav:
                st.session_state.pending_voice = transcript
                st.session_state.voice_label   = "Sending to Chat…"
            st.rerun()
        else:
            st.session_state.voice_status = "idle"
            st.session_state.voice_label  = ""

# ── Guard ─────────────────────────────────────────────────────────────────────
if not st.session_state.ready:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem 2rem;text-align:center;">
        <div style="width:64px;height:64px;border-radius:20px;background:linear-gradient(135deg,#6366f1,#a855f7);display:flex;align-items:center;justify-content:center;font-size:28px;margin-bottom:20px;box-shadow:0 8px 24px rgba(99,102,241,0.25);">📄</div>
        <div style="font-size:18px;font-weight:700;color:var(--text-primary);margin-bottom:8px;">Get Started</div>
        <div style="font-size:14px;color:var(--text-muted);max-width:400px;line-height:1.6;">
            Upload your bank statement PDF in the sidebar to unlock AI-powered financial insights, spending analysis, and budget planning.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

current_page = TABS[st.session_state.active_tab]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CHAT ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
if current_page == "💬 Chat Advisor":
    st.markdown("<h2 style='font-size: 24px; font-weight: 800; margin-bottom: 4px;'>💬 AI Financial Advisor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 14px; margin-bottom: 1.5rem;'>Ask anything about your transactions — typing or voice both work.</p>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Let Streamlit natively escape HTML (prevents </div> from breaking DOM)
            st.markdown(msg["content"])

    pending = st.session_state.pop("pending_voice", None)
    typed   = st.chat_input("Ask about your transactions…")
    prompt  = pending or typed

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):

                if not st.session_state.get("db"):
                    st.markdown("""
                    <div style="padding:16px;border-radius:12px;background:var(--red-subtle);border:1px solid var(--red);color:var(--text-primary);font-size:14px;">
                        <b>❌ Invalid PDF Detected</b><br><br>
                        This file does not contain readable text.<br>
                        Please upload a proper bank statement (not scanned or image-based).
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()

                response = get_finance_advice(prompt, st.session_state.db)
                response = strip_html_tags(response)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FULL AUDIT REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "📊 Full Audit Report":
    st.markdown("<h2 style='font-size: 24px; font-weight: 800; margin-bottom: 4px;'>📊 Full Statement Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 14px; margin-bottom: 1.5rem;'>Comprehensive audit of your transactions, balances, spending patterns, and AI-powered security alerts.</p>", unsafe_allow_html=True)

    if "report" in st.session_state:
        st.markdown("<div style='background:var(--green-subtle);border:1px solid var(--green);border-radius:12px;padding:12px 16px;font-size:13px;color:var(--green);font-weight:600;margin-bottom:1rem;'>✅ Report already generated. Click below to regenerate.</div>", unsafe_allow_html=True)

    if st.button("🔍 Generate Full Audit Report", type="primary"):
        try:
            with st.spinner("Extracting every detail… this may take 30–60 seconds."):
                report = get_detailed_report(
                    st.session_state.opening_balance, st.session_state.closing_balance,
                    st.session_state.first_page_text, st.session_state.raw_docs,
                )
                st.session_state.report = report
                st.session_state.pop("anomalies_text", None)


                capture_eval_snapshot(
                    report, 
                    st.session_state.db, 
                    st.session_state.last_run_metrics
                )




        except Exception as e:
            st.error(f"Error generating report: {e}")
            st.stop()

    if "report" in st.session_state:
        report = st.session_state.report

        # Account Info Cards
        st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 17px; font-weight: 700; margin-bottom: 1rem;'>🏦 Account Information</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="card">
                <div style="font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; font-weight: 600;">Account Holder</div>
                <div style="font-size: 15px; font-weight: 700; color: var(--text-primary); margin-bottom: 12px;">{report.account_info.customer_name}</div>
                <div style="font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; font-weight: 600;">Account Number</div>
                <div style="font-size: 14px; font-weight: 600; color: var(--text-primary); font-family: var(--mono);">{report.account_info.account_number}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card">
                <div style="font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; font-weight: 600;">IFSC Code</div>
                <div style="font-size: 14px; font-weight: 600; color: var(--text-primary); font-family: var(--mono); margin-bottom: 12px;">{report.account_info.ifsc_code}</div>
                <div style="font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; font-weight: 600;">Statement Period</div>
                <div style="font-size: 14px; font-weight: 600; color: var(--text-primary);">{report.account_info.statement_period}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("💰", "purple", "Opening Balance", f"₹{st.session_state.opening_balance:,.2f}")
        with m2:
            render_metric_card("📤", "red", "Total Debits", f"₹{report.total_debits:,.2f}")
        with m3:
            render_metric_card("📥", "green", "Total Credits", f"₹{report.total_credits:,.2f}")
        with m4:
            render_metric_card("🏦", "blue", "Closing Balance", f"₹{st.session_state.closing_balance:,.2f}")

        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

        st.markdown("<h3 style='font-size: 17px; font-weight: 700; margin-bottom: 1rem;'>📑 Transaction History</h3>", unsafe_allow_html=True)
        df_txn = pd.DataFrame([t.model_dump() for t in report.transactions])
        st.dataframe(df_txn, use_container_width=True, height=320)

        # Spending Breakdown + AI Alerts side by side
        col_chart, col_alerts = st.columns([3, 2])
        with col_chart:
            st.markdown("<h3 style='font-size: 17px; font-weight: 700; margin-bottom: 1rem;'>📊 Spending Breakdown</h3>", unsafe_allow_html=True)
            spending_df = df_txn[df_txn["debit"] > 0].copy()
            fig = None
            if not spending_df.empty:
                chart_data = spending_df.groupby("category")["debit"].sum().reset_index()
                fig = px.pie(chart_data, values="debit", names="category",
                             hole=0.55, color_discrete_sequence=_COLOR_PALETTE)
                fig = update_plotly_layout(fig)
                fig.update_layout(
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=-0.15, font=dict(size=11)),
                    margin=dict(t=20, b=20, l=80, r=20),
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("""
                <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:2rem;text-align:center;">
                    <div style="width:48px;height:48px;border-radius:14px;background:var(--surface-2);border:1px solid var(--border);display:flex;align-items:center;justify-content:center;font-size:20px;margin-bottom:12px;">📉</div>
                    <div style="font-size:14px;font-weight:600;color:var(--text-secondary);margin-bottom:4px;">No Spending Data</div>
                    <div style="font-size:12px;color:var(--text-muted);max-width:300px;line-height:1.5;">No debit transactions found in this statement to visualize.</div>
                </div>
                """, unsafe_allow_html=True)

        with col_alerts:
            st.markdown("<h3 style='font-size: 17px; font-weight: 700; margin-bottom: 1rem;'>🚩 AI Security Alerts</h3>", unsafe_allow_html=True)
            if "anomalies_text" not in st.session_state:
                with st.spinner("Scanning for anomalies…"):
                    st.session_state.anomalies_text = get_finance_advice(
                        "List all suspicious, unusually large, or duplicate transactions "
                        "as bullet points with dates and amounts.",
                        st.session_state.db,
                    )
            render_ai_insight(st.session_state.anomalies_text)

        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

        col_pdf, _ = st.columns([1, 3])
        with col_pdf:
            try:
                pdf_buf = generate_pdf_report(
                    report, fig, st.session_state.get("anomalies_text")
                )
                st.download_button(
                    "📥 Download Full Report PDF", pdf_buf,
                    f"Report_{report.account_info.customer_name}.pdf",
                    "application/pdf", use_container_width=True,
                )
            except Exception as pdf_err:
                st.error(f"PDF error: {pdf_err}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BUDGET PLANNER
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "🎯 Budget Planner":
    st.markdown("<h2 style='font-size: 24px; font-weight: 800; margin-bottom: 4px;'>🎯 Smart Budget Planner</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 14px; margin-bottom: 1.5rem;'>Set spending goals and track your progress by category.</p>", unsafe_allow_html=True)

    if "report" not in st.session_state:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3rem 2rem;text-align:center;">
            <div style="width:56px;height:56px;border-radius:16px;background:var(--amber-subtle);border:1px solid var(--amber);display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:16px;">📊</div>
            <div style="font-size:16px;font-weight:700;color:var(--text-primary);margin-bottom:6px;">Report Required</div>
            <div style="font-size:13px;color:var(--text-muted);max-width:360px;line-height:1.6;">Generate the Full Audit Report first to unlock budget planning and spending insights.</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    report = st.session_state.report

    df = pd.DataFrame([t.model_dump() for t in report.transactions])
    df["date_dt"]    = parse_dates_flexible(df["txn_date"])
    df["month_year"] = df["date_dt"].dt.strftime("%B %Y")

    available_months = [m for m in df["month_year"].dropna().unique().tolist() if m]
    if not available_months:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:2rem;text-align:center;">
            <div style="width:48px;height:48px;border-radius:14px;background:var(--red-subtle);border:1px solid var(--red);display:flex;align-items:center;justify-content:center;font-size:20px;margin-bottom:12px;">⚠️</div>
            <div style="font-size:14px;font-weight:600;color:var(--text-secondary);margin-bottom:4px;">Date Parsing Error</div>
            <div style="font-size:12px;color:var(--text-muted);max-width:320px;line-height:1.5;">Could not parse transaction dates from this statement. Please check the file format and try again.</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    selected_month = st.selectbox("📅 Select Month to Analyse", available_months)

    month_df      = df[df["month_year"] == selected_month].copy()
    spending_df   = df[df["debit"] > 0].copy()
    all_cats      = sorted(spending_df["category"].dropna().unique().tolist())
    extra_cats    = [c for c in all_cats if c not in CORE_CATS]

    show_all    = st.toggle(
        f"🔍 View All Categories ({len(all_cats)} found in statement)",
        value=st.session_state.get("budget_show_all", False),
        key="budget_show_all",
    )
    active_cats = all_cats if show_all else [c for c in CORE_CATS if c in all_cats]

    if not show_all and extra_cats:
        st.caption(f"Hidden: {', '.join(extra_cats)} — toggle to set goals for all.")

    month_spend_df = month_df[month_df["debit"] > 0]
    for cat in all_cats:
        if cat not in st.session_state.budgets:
            actual    = float(month_spend_df[month_spend_df["category"] == cat]["debit"].sum())
            suggested = max(500, round(actual * 1.2 / 500) * 500) if actual > 0 else 2000
            st.session_state.budgets[cat] = suggested

    with st.form("budget_form"):
        st.markdown(f"<h3 style='font-size: 17px; font-weight: 700; margin-bottom: 1rem;'>Set Your Goals for {selected_month}</h3>", unsafe_allow_html=True)
        user_goals: dict = {}

        if show_all and extra_cats:
            core_visible = [c for c in CORE_CATS if c in all_cats]
            st.markdown("**📌 Core Categories**")
            cols = st.columns(2)
            for i, cat in enumerate(core_visible):
                user_goals[cat] = cols[i % 2].number_input(
                    f"{cat} (₹)", min_value=0,
                    value=int(st.session_state.budgets.get(cat, 2000)),
                    step=500, key=f"bg_{cat}",
                )
            st.markdown("**📂 Additional Categories**")
            cols2 = st.columns(2)
            for i, cat in enumerate(extra_cats):
                user_goals[cat] = cols2[i % 2].number_input(
                    f"{cat} (₹)", min_value=0,
                    value=int(st.session_state.budgets.get(cat, 2000)),
                    step=500, key=f"bg_{cat}",
                )
        else:
            cols = st.columns(2)
            for i, cat in enumerate(active_cats):
                user_goals[cat] = cols[i % 2].number_input(
                    f"{cat} (₹)", min_value=0,
                    value=int(st.session_state.budgets.get(cat, 2000)),
                    step=500, key=f"bg_{cat}",
                )

        submitted = st.form_submit_button("🔥 Check My Budget", type="primary")

    if submitted:
        st.session_state.budgets.update(user_goals)
        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='font-size: 17px; font-weight: 700; margin-bottom: 1rem;'>📊 Budget vs Actual — {selected_month}</h3>", unsafe_allow_html=True)

        over_count = under_count = 0
        for cat, goal in user_goals.items():
            actual  = float(month_df[month_df["category"] == cat]["debit"].sum())
            diff    = goal - actual
            percent = min(actual / goal, 1.0) if goal > 0 else 0.0

            if diff < 0:
                over_count += 1
            else:
                under_count += 1

            icon_emoji = txn_icon_emoji(cat)
            icon_color = txn_icon_color(cat)
            with st.container():
                st.markdown(f"""
                <div class="budget-card" style="margin-bottom: 12px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
                        <div style="display:flex;align-items:center;gap:10px;">
                            <div class="metric-icon-bg {icon_color}" style="width:32px;height:32px;font-size:14px;">{icon_emoji}</div>
                            <div>
                                <div style="font-size:14px;font-weight:700;color:var(--text-primary);">{cat}</div>
                                <div style="font-size:12px;color:var(--text-muted);">Spent ₹{actual:,.0f} · Goal ₹{goal:,.0f}</div>
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:14px;font-weight:700;color:{'var(--green)' if diff >= 0 else 'var(--red)'};">{'✅ Under' if diff >= 0 else '🚨 Over'}</div>
                            <div style="font-size:12px;color:var(--text-muted);font-family:var(--mono);">₹{abs(diff):,.0f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(percent)
            st.markdown("<div style='margin: 8px 0;'></div>", unsafe_allow_html=True)

        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            render_metric_card("✅", "green", "Under Budget", str(under_count))
        with c2:
            render_metric_card("🚨", "red", "Over Budget", str(over_count), f"{over_count} need attention" if over_count else "All good!", "negative" if over_count else "positive")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SPENDING FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "🔮 Spending Forecast":
    st.markdown("<h2 style='font-size: 24px; font-weight: 800; margin-bottom: 4px;'>🔮 AI Spending Forecast</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 14px; margin-bottom: 1.5rem;'>Predict future spending patterns based on your transaction history.</p>", unsafe_allow_html=True)

    if "report" in st.session_state:
        data_pack, error = calculate_forecast(st.session_state.report.transactions)
        if error:
            st.warning(error)
        else:
            monthly_df, next_date, pred_val = data_pack
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_df["date_dt"], y=monthly_df["debit"],
                name="Actual", mode="lines+markers",
                line=dict(color="#6366f1", width=3),
                marker=dict(size=8, color="#6366f1", line=dict(color="rgba(99,102,241,0.3)", width=2)),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
            ))
            fig.add_trace(go.Scatter(
                x=[monthly_df["date_dt"].iloc[-1], next_date],
                y=[monthly_df["debit"].iloc[-1], pred_val],
                name="Forecast", mode="lines+markers",
                line=dict(dash="dot", color="#a855f7", width=3),
                marker=dict(size=10, color="#a855f7", symbol="diamond"),
            ))
            fig = update_plotly_layout(fig)
            fig.update_layout(
                margin=dict(t=20, b=40, l=60, r=20),
                hovermode="x unified",
                xaxis_title="Month",
                yaxis_title="Amount (₹)",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.spinner("AI Analysis…"):
                insights = get_forecast_insights(monthly_df, pred_val)
                render_ai_insight(f"<b>Trend Analysis:</b> {insights.trend_analysis}")
    else:
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3rem 2rem;text-align:center;">
            <div style="width:56px;height:56px;border-radius:16px;background:var(--blue-subtle);border:1px solid var(--blue);display:flex;align-items:center;justify-content:center;font-size:24px;margin-bottom:16px;">🔮</div>
            <div style="font-size:16px;font-weight:700;color:var(--text-primary);margin-bottom:6px;">Forecast Unavailable</div>
            <div style="font-size:13px;color:var(--text-muted);max-width:360px;line-height:1.6;">Generate the Full Audit Report first to unlock AI-powered spending forecasts and trend analysis.</div>
        </div>
        """, unsafe_allow_html=True)