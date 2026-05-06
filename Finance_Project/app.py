import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.advicor import (calculate_forecast,
                              get_detailed_report, get_finance_advice,
                              get_forecast_insights, generate_pdf_report)
from modules.processor import process_pdf_to_memory
from modules.voice import (classify_intent, get_audio_hash, normalize_transcript, transcribe_audio)






def inject_styles():
    st.markdown(
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

/* ── Design Tokens ── */
:root {
    --blue:        #3b82f6;
    --blue-light:  #eff6ff;
    --blue-mid:    rgba(59,130,246,0.12);
    --blue-dark:   #1d4ed8;
    --green:       #10b981;
    --green-light: #ecfdf5;
    --red:         #ef4444;
    --red-light:   #fef2f2;
    --amber:       #f59e0b;
    --bg:          #f1f5f9;
    --surface:     #ffffff;
    --sidebar-bg:  #0f172a;
    --text:        #0f172a;
    --text2:       #64748b;
    --text3:       #94a3b8;
    --border:      #e2e8f0;
    --radius:      14px;
    --sidebar-w:   248px;
    --shadow-sm:   0 1px 4px rgba(15,23,42,0.06);
    --shadow-md:   0 4px 16px rgba(15,23,42,0.1);
}

/* ── Global ── */
* { font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important; }
h1, h2, h3, h4 { color: var(--text) !important; font-weight: 600 !important; letter-spacing: -0.01em !important; }
p, span, li { color: var(--text2) !important; }
body { background: var(--bg) !important; }

/* ── App Shell ── */
[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
}
[data-testid="stMainBlockContainer"] {
    padding: 2rem 2.5rem !important;
    max-width: 1200px !important;
}
[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }

/* ── Sidebar: Dark Premium ── */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    min-width: var(--sidebar-w) !important;
    max-width: var(--sidebar-w) !important;
    transition: min-width 0.28s cubic-bezier(.4,0,.2,1),
                max-width 0.28s cubic-bezier(.4,0,.2,1) !important;
}

/* Sidebar text globally */
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #475569 !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin: 1.25rem 0 0.5rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    font-size: 13.5px !important;
    color: #94a3b8 !important;
    line-height: 1.5 !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    margin: 1rem 0 !important;
}

/* Sidebar navigation radio */
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 9px 12px !important;
    border-radius: 9px !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
    transition: all 0.15s ease !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    margin: 1px 6px !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] .stRadio label[data-checked="true"],
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(59,130,246,0.18) !important;
    border-color: rgba(59,130,246,0.25) !important;
    color: #93c5fd !important;
}

/* Sidebar file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    margin: 6px 0 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px dashed rgba(59,130,246,0.3) !important;
    border-radius: 10px !important;
    transition: border-color 0.15s, background 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] > div:hover {
    border-color: rgba(59,130,246,0.55) !important;
    background: rgba(59,130,246,0.06) !important;
}

/* Sidebar collapse button — show it properly */
[data-testid="stSidebarCollapsedControl"],
button[data-testid="collapsedControl"],
[data-testid="stSidebarNav"] button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #64748b !important;
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapsedControl"]:hover {
    background: rgba(255,255,255,0.14) !important;
    color: #94a3b8 !important;
}

/* ── Topbar / Header area ── */
[data-testid="stAppViewContainer"] > [data-testid="stMainBlockContainer"] > div:first-child h1 {
    font-size: 28px !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--text) !important;
    margin-bottom: 4px !important;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px 20px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow 0.15s, transform 0.15s !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    font-size: 12px !important;
    color: var(--text3) !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 500 !important;
    color: var(--text) !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 14px 16px !important;
    box-shadow: var(--shadow-sm) !important;
    margin-bottom: 10px !important;
    animation: fadeUp 0.25s ease-out !important;
    max-width: 80% !important;
}
/* User messages — right side */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(59,130,246,0.08) !important;
    border-color: rgba(59,130,246,0.18) !important;
    margin-left: auto !important;
    margin-right: 0 !important;
}
/* AI messages — left side */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    margin-right: auto !important;
    margin-left: 0 !important;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Chat Input ── */
[data-testid="stChatInputContainer"] {
    padding: 12px 0 0 !important;
}
[data-testid="stChatInputContainer"] textarea {
    border-radius: 12px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--surface) !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    resize: none !important;
    color: var(--text) !important;
}
[data-testid="stChatInputContainer"] textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
    outline: none !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button,
[data-testid="stDownloadButton"] > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 9px 18px !important;
    transition: all 0.15s cubic-bezier(.4,0,.2,1) !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
}
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stDownloadButton"] > button {
    background: var(--blue) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(59,130,246,0.28) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover,
[data-testid="stDownloadButton"] > button:hover {
    background: var(--blue-dark) !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.38) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:not([kind="primary"]) {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stButton"] > button:not([kind="primary"]):hover {
    border-color: var(--blue) !important;
    color: var(--blue-dark) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}

/* ── Alerts / Info boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    padding: 12px 16px !important;
    border-left: 3px solid !important;
    font-size: 13.5px !important;
}
[data-testid="stAlert"][data-baseweb="notification"] {
    border-color: var(--amber) !important;
    background: #fffbeb !important;
}
[data-testid="stAlert"][data-baseweb="notification"] * { color: #92400e !important; }
.stSuccess { border-color: var(--green) !important; background: var(--green-light) !important; }
.stSuccess * { color: #065f46 !important; }
.stError { border-color: var(--red) !important; background: var(--red-light) !important; }
.stError * { color: #991b1b !important; }

/* st.info() */
div[data-testid="stAlertContainer"] > div[role="alert"] {
    border-radius: 10px !important;
    border-left: 3px solid var(--blue) !important;
    background: var(--blue-light) !important;
    padding: 12px 16px !important;
}
div[data-testid="stAlertContainer"] > div[role="alert"] * { color: var(--blue-dark) !important; }

/* ── Progress Bars ── */
[data-testid="stProgress"] > div {
    background: #e2e8f0 !important;
    border-radius: 8px !important;
    height: 8px !important;
    overflow: hidden !important;
}
[data-testid="stProgress"] > div > div {
    border-radius: 8px !important;
    height: 100% !important;
    background: linear-gradient(90deg, var(--blue), var(--green)) !important;
    transition: width 0.4s ease !important;
}

/* ── Data Table ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] thead th {
    background: #f8fafc !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: var(--text2) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] tbody td {
    font-size: 13.5px !important;
    padding: 10px 14px !important;
    color: var(--text) !important;
    border-bottom: 1px solid #f1f5f9 !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: #f8fafc !important;
}

/* ── Selectbox / Dropdowns ── */
[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--surface) !important;
    padding: 8px 14px !important;
    font-size: 14px !important;
    transition: border-color 0.15s !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.08) !important;
}

/* ── Number Input ── */
[data-testid="stNumberInput"] input {
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--surface) !important;
    padding: 9px 14px !important;
    font-size: 14px !important;
    font-family: 'DM Mono', monospace !important;
    color: var(--text) !important;
    transition: border-color 0.15s !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.08) !important;
    outline: none !important;
}

/* ── Toggle ── */
[data-testid="stToggle"] label { font-size: 13.5px !important; color: var(--text2) !important; }

/* ── Forms ── */
[data-testid="stForm"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px 22px !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] > summary {
    padding: 12px 16px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    background: var(--surface) !important;
}
[data-testid="stExpander"] > summary:hover { background: #f8fafc !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--blue) !important; }

/* ── Voice / Audio Input ── */
[data-testid="stAudioInput"] {
    display: flex !important;
    justify-content: center !important;
    margin: 8px 0 !important;
}
[data-testid="stAudioInput"] button {
    width: 52px !important;
    height: 52px !important;
    border-radius: 50% !important;
    background: var(--green) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(16,185,129,0.32) !important;
    font-size: 20px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stAudioInput"] button:hover {
    transform: scale(1.08) !important;
    box-shadow: 0 6px 20px rgba(16,185,129,0.42) !important;
}
[data-testid="stAudioInput"] button[data-recording="true"] {
    animation: micPulse 1.4s ease-in-out infinite !important;
}
@keyframes micPulse {
    0%, 100% { box-shadow: 0 4px 14px rgba(16,185,129,0.32); }
    50%       { box-shadow: 0 4px 28px rgba(16,185,129,0.62); }
}

/* ── Caption / Subtitle ── */
[data-testid="stCaptionContainer"] p,
.stCaption {
    font-size: 14px !important;
    color: var(--text3) !important;
}

/* ── Subheader ── */
h2[class*="stSubheader"] { font-size: 18px !important; }
h3[class*="stSubheader"] { font-size: 16px !important; }

/* ── Plotly Charts ── */
.js-plotly-plot { border-radius: 12px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(100,116,139,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(100,116,139,0.35); }

/* ── Print ── */
@media print {
    [data-testid="stSidebar"],
    [data-testid="stToolbar"],
    [data-testid="stDeployButton"],
    .stButton, .no-print { display: none !important; }
    body, [data-testid="stAppViewContainer"] {
        background: white !important;
    }
    [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
}

/* ── Responsive: Narrower screens ── */
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding: 1rem !important; }
    [data-testid="stSidebar"] { min-width: 200px !important; max-width: 200px !important; }
}
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding: 0.75rem !important; }
}
</style>
"""
, unsafe_allow_html=True)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Finance Advisor", page_icon="💰", layout="wide")

# ── CSS — Premium Fintech Dashboard (Stripe/Razorpay/CRED style) ──────────────


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


# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "ready": False, "messages": [], "opening_balance": 0.0, "closing_balance": 0.0,
    "active_tab": 0, "voice_nav": None, "last_voice_hash": "",
    "voice_status": "idle", "voice_label": "",
    "budgets": {"Food & Dining": 5000, "Travel & Transport": 3000,
                "Shopping": 4000, "Utilities & Bills": 2000},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Apply voice navigation before sidebar renders
if st.session_state.get("voice_nav") is not None:
    st.session_state.active_tab = st.session_state.voice_nav
    st.session_state.voice_nav  = None

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💰 AI Personal Finance Advisor")
st.caption("Upload your bank statement · ask questions · track spending · forecast the future")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
voice_file = None
with st.sidebar:
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
                db, opening, closing, first_page, raw_docs = process_pdf_to_memory(path)
                st.session_state.update({
                    "db": db, "opening_balance": opening, "closing_balance": closing,
                    "first_page_text": first_page, "raw_docs": raw_docs, "ready": True,
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
    st.info("👈 Upload a bank statement PDF in the sidebar to get started.")
    st.stop()

current_page = TABS[st.session_state.active_tab]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CHAT ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
if current_page == "💬 Chat Advisor":
    st.subheader("💬 AI Financial Advisor")
    st.caption("Ask anything about your transactions — typing or voice both work.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
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
                response = get_finance_advice(prompt, st.session_state.db)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FULL AUDIT REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "📊 Full Audit Report":
    st.header("📊 Full Statement Analysis")

    if "report" in st.session_state:
        st.info("✅ Report already generated. Click below to regenerate.")

    st.markdown('<div class="no-print">', unsafe_allow_html=True)
    if st.button("🔍 Generate Full Audit Report", type="primary"):
        try:
            with st.spinner("Extracting every detail… this may take 30–60 seconds."):
                report = get_detailed_report(
                    st.session_state.opening_balance, st.session_state.closing_balance,
                    st.session_state.first_page_text, st.session_state.raw_docs,
                )
                st.session_state.report = report
                st.session_state.pop("anomalies_text", None)
        except Exception as e:
            st.error(f"Error generating report: {e}")
            st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

    if "report" in st.session_state:
        report = st.session_state.report

        st.subheader("🏦 Account Information")
        c1, c2 = st.columns(2)
        with c1:
            st.info(
                f"**Customer:** {report.account_info.customer_name}\n\n"
                f"**A/C No:** {report.account_info.account_number}\n\n"
                f"**Account Type:** {report.account_info.account_type}"
            )
        with c2:
            st.info(
                f"**IFSC:** {report.account_info.ifsc_code}\n\n"
                f"**Branch:** {report.account_info.branch_name}\n\n"
                f"**Period:** {report.account_info.statement_period}"
            )
        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Opening Balance", f"₹{st.session_state.opening_balance:,.2f}")
        m2.metric("Total Debits",    f"₹{report.total_debits:,.2f}")
        m3.metric("Total Credits",   f"₹{report.total_credits:,.2f}")
        m4.metric("Closing Balance", f"₹{st.session_state.closing_balance:,.2f}")
        st.divider()

        st.markdown('<div class="no-print">', unsafe_allow_html=True)
        st.subheader("📑 Transaction History")
        df_txn = pd.DataFrame([t.model_dump() for t in report.transactions])
        st.dataframe(df_txn, use_container_width=True, height=300)
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("📊 Spending Breakdown")
        spending_df = df_txn[df_txn["debit"] > 0].copy()
        fig = None
        if not spending_df.empty:
            chart_data = spending_df.groupby("category")["debit"].sum().reset_index()
            fig = px.pie(chart_data, values="debit", names="category",
                         hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h", yanchor="bottom", y=-0.3))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No debit transactions found.")
        st.divider()

        st.subheader("🚩 AI Security Alerts")
        if "anomalies_text" not in st.session_state:
            with st.spinner("Scanning for anomalies…"):
                st.session_state.anomalies_text = get_finance_advice(
                    "List all suspicious, unusually large, or duplicate transactions "
                    "as bullet points with dates and amounts.",
                    st.session_state.db,
                )
        st.warning(st.session_state.anomalies_text)
        st.divider()

        st.markdown('<div class="no-print">', unsafe_allow_html=True)
        col_pdf, col_print = st.columns(2)
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
        with col_print:
            st.markdown(
                '<button onclick="window.print()" style="width:100%;height:45px;'
                'background:linear-gradient(135deg,#10b981,#059669);color:white;border:none;'
                'border-radius:10px;cursor:pointer;font-weight:600;">🖨️ Print View</button>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
        st.success("✅ Analysis complete!")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BUDGET PLANNER
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "🎯 Budget Planner":
    st.header("🎯 Smart Budget Planner")

    if "report" not in st.session_state:
        st.warning("⚠️ Please generate the Full Audit Report first (Page 2).")
        st.stop()

    report = st.session_state.report

    # Build df with FLEXIBLE date parsing — fixes the all-zero bug
    df = pd.DataFrame([t.model_dump() for t in report.transactions])
    df["date_dt"]    = parse_dates_flexible(df["txn_date"])
    df["month_year"] = df["date_dt"].dt.strftime("%B %Y")

    available_months = [m for m in df["month_year"].dropna().unique().tolist() if m]
    if not available_months:
        st.error("Could not parse transaction dates. Please check statement format.")
        st.stop()

    selected_month = st.selectbox("📅 Select Month to Analyse", available_months)

    # This is the KEY fix: filter BEFORE computing actuals
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

    # Seed defaults from ACTUAL monthly spend (not zeros)
    month_spend_df = month_df[month_df["debit"] > 0]
    for cat in all_cats:
        if cat not in st.session_state.budgets:
            actual    = float(month_spend_df[month_spend_df["category"] == cat]["debit"].sum())
            suggested = max(500, round(actual * 1.2 / 500) * 500) if actual > 0 else 2000
            st.session_state.budgets[cat] = suggested

    with st.form("budget_form"):
        st.subheader(f"Set Your Goals for {selected_month}")
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
        st.divider()
        st.subheader(f"📊 Budget vs Actual — {selected_month}")

        over_count = under_count = 0
        for cat, goal in user_goals.items():
            # Use month_df filtered to selected month — actual real spend
            actual  = float(month_df[month_df["category"] == cat]["debit"].sum())
            diff    = goal - actual
            percent = min(actual / goal, 1.0) if goal > 0 else 0.0

            if diff < 0:
                over_count += 1
            else:
                under_count += 1

            with st.container():
                col_l, col_r = st.columns([3, 1])
                with col_l:
                    st.markdown(f"**{cat}**")
                    st.caption(f"Spent ₹{actual:,.0f}  ·  Goal ₹{goal:,.0f}")
                    st.progress(percent)
                with col_r:
                    if diff >= 0:
                        st.success(f"✅ Under\n₹{diff:,.0f}")
                    else:
                        st.error(f"🚨 Over\n₹{abs(diff):,.0f}")
            st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("✅ Under Budget", under_count)
        with c2:
            st.metric("🚨 Over Budget", over_count,
                      delta=f"{over_count} need attention" if over_count else "All good!",
                      delta_color="inverse" if over_count else "normal")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SPENDING FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "🔮 Spending Forecast":
    st.header("🔮 AI Spending Forecast")
    if "report" in st.session_state:
        data_pack, error = calculate_forecast(st.session_state.report.transactions)
        if error:
            st.warning(error)
        else:
            monthly_df, next_date, pred_val = data_pack
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_df["date_dt"], y=monthly_df["debit"],
                                     name="Actual", line=dict(color="#6366f1", width=3)))
            fig.add_trace(go.Scatter(
                x=[monthly_df["date_dt"].iloc[-1], next_date],
                y=[monthly_df["debit"].iloc[-1], pred_val],
                name="Forecast", line=dict(dash="dot", color="#a855f7", width=3),
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            with st.spinner("AI Analysis…"):
                insights = get_forecast_insights(monthly_df, pred_val)
                st.info(f"**Trend Analysis:** {insights.trend_analysis}")
    else:
        st.info("👈 Generate the Full Audit Report first.")