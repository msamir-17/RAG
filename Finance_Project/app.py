import hashlib
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from modules.advicor import (calculate_forecast, generate_pdf_report,
                              get_detailed_report, get_finance_advice,
                              get_forecast_insights)
from modules.processor import process_pdf_to_memory
from modules.voice import classify_intent , transcribe_audio , get_audio_hash

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Finance Advisor", page_icon="💰", layout="wide")

# ── Theme-aware CSS (works in both dark and light mode) ───────────────────────
st.markdown("""
<style>
/* ── Adaptive card — works in dark and light ── */
.fin-card {
    background: var(--background-color, #ffffff);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 16px !important;
}

/* ── Sidebar nav radio — cleaner look ── */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}
[data-testid="stSidebar"] .stRadio > div > label {
    border-radius: 8px !important;
    padding: 8px 12px !important;
    transition: background 0.15s ease !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(99,102,241,0.12) !important;
}

/* ── Progress bar color ── */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6366f1, #a855f7) !important;
    border-radius: 99px !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    border: 1px solid rgba(128,128,128,0.15) !important;
    margin-bottom: 8px !important;
}

/* ── Voice recorder button ── */
[data-testid="stAudioInput"] button {
    border-radius: 99px !important;
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    color: white !important;
    border: none !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* ── Generate button ── */
[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
}

/* ── Selectbox and number input ── */
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] > div {
    border-radius: 8px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_audio_hash(audio_bytes: bytes) -> str:
    return hashlib.md5(audio_bytes).hexdigest()


def route_intent(text: str) -> int:
    """
    FIX: case-insensitive, handles partial matches and natural phrases.
    'generate the audit report' → 1
    'show me forecast'          → 3
    'open budget planner'       → 2
    """
    t = text.lower()

    audit_words    = ["audit", "full report", "full statement", "generate report",
                      "statement analysis", "transaction history"]
    budget_words   = ["budget", "goal", "planner", "spending limit", "set limit"]
    forecast_words = ["forecast", "predict", "next month", "future spending",
                      "trend", "projection"]

    if any(w in t for w in audit_words):    return 1
    if any(w in t for w in budget_words):   return 2
    if any(w in t for w in forecast_words): return 3
    return 0  # Default → Chat


TABS = ["💬 Chat Advisor", "📊 Full Audit Report", "🎯 Budget Planner", "🔮 Spending Forecast"]

# ── Session State Defaults ────────────────────────────────────────────────────
_defaults = {
    "ready":            False,
    "messages":         [],
    "opening_balance":  0.0,
    "closing_balance":  0.0,
    "active_tab":       0,
    "last_audio_hash":  "",
    "budgets": {
        "Food & Dining":       5000,
        "Travel & Transport":  3000,
        "Shopping":            4000,
        "Utilities & Bills":   2000,
    },
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💰 AI Personal Finance Advisor")
st.caption("Upload your bank statement · ask questions · track spending · forecast the future")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Center")
    uploaded_file = st.file_uploader("Choose a bank statement (PDF)", type="pdf")

    if uploaded_file:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_uploaded_file") != file_id:
            
            # 1. WIPE OLD DATA
            for key in list(st.session_state.keys()):
                if key not in ["budgets", "active_tab"]: 
                    del st.session_state[key]
            
            # 2. RE-INITIALIZE DEFAULTS (So the app doesn't crash)
            st.session_state.ready = False
            st.session_state.messages = []
            st.session_state.last_uploaded_file = file_id
            
            # 3. SAVE AND PROCESS
            os.makedirs("data", exist_ok=True)
            file_path = os.path.join("data", "temp_statement.pdf")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with st.spinner("Reading new statement..."):
                # Now processing starts from a clean memory
                db, opening, closing, first_page, raw_docs = process_pdf_to_memory(file_path)
                st.session_state.db = db
                st.session_state.opening_balance = opening
                st.session_state.closing_balance = closing
                st.session_state.first_page_text = first_page
                st.session_state.raw_docs = raw_docs
                st.session_state.ready = True
                st.success("✅ New statement loaded!")
    # ── Navigation (voice can change active_tab before this renders) ──
    if st.session_state.ready:
        st.divider()
        st.caption("NAVIGATE")
        selection = st.radio(
            "Page",
            TABS,
            index=st.session_state.active_tab,
            key="nav_radio",
            label_visibility="collapsed",
        )
        # Keep active_tab in sync with manual clicks
        if TABS.index(selection) != st.session_state.active_tab:
            st.session_state.active_tab = TABS.index(selection)


if st.session_state.ready:
    with st.sidebar:
        st.divider()
        st.caption("🎤 VOICE COMMAND")
        st.markdown(
            "<small>Say <b>'generate audit report'</b>, <b>'show forecast'</b>, "
            "<b>'open budget'</b>, or ask any question.</small>",
            unsafe_allow_html=True,
        )
        voice_file = st.audio_input("Record", key="global_voice", label_visibility="collapsed")

    # FIX: read bytes ONCE, hash it, only process if new recording
    
         # 1. Use getvalue() - NEVER use .read() here
    # Place this at the VERY START of the 'if st.session_state.ready' block
    if voice_file:
        # 1. Capture the raw data safely
        # audio_data = voice_file.getvalue()
        audio_data = voice_file.getvalue()  
        
        # 2. Calculate the fingerprint
        current_voice_hash = get_audio_hash(audio_data)

        # 3. THE GUARD: Only proceed if this hash is DIFFERENT from the last one
        if st.session_state.get("last_voice_hash") != current_voice_hash:
            # Mark as processed immediately to prevent re-entry during rerun
            st.session_state.last_voice_hash = current_voice_hash
            
            # Now it is safe to proceed to Step 2 (Transcribe)
            with st.spinner("🎙️ Processing unique voice command..."):
                transcript = transcribe_audio(audio_data)

                if transcript:
                    from modules.voice import normalize_transcript
                    transcript = normalize_transcript(transcript)

                    # Save to chat
                    st.session_state.messages.append({
                        "role": "user",
                        "content": transcript
                    })

                    # Get AI response
                    response = get_finance_advice(transcript, st.session_state.db)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

                    st.write(response)
                else:
                    st.error("❌ Could not understand audio")
        else:
            # This is a rerun, we already handled this audio. Do nothing.
            pass
# ── Guard: nothing to show if no file uploaded ────────────────────────────────
if not st.session_state.ready:
    st.info("👈 Upload a bank statement in the sidebar to get started.")
    st.image("https://cdn-icons-png.flaticon.com/512/1611/1611179.png", width=90)
    st.stop()

current_page = TABS[st.session_state.active_tab]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CHAT ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
if current_page == "💬 Chat Advisor":
    st.subheader("💬 AI Financial Advisor")
    st.caption("Ask anything about your transactions — typing or voice both work.")

    # Render history
    # 1. Check if a voice message was sent from the sidebar
    pending_prompt = st.session_state.pop("pending_voice", None)

    # 2. Get the typed input
    typed_prompt = st.chat_input("Ask about your transactions…")

    # 3. Use whichever one exists
    final_prompt = pending_prompt or typed_prompt

    # Pick up voice prompt if navigation didn't consume it
   

    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.markdown(final_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                response = get_finance_advice(final_prompt, st.session_state.db)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — FULL AUDIT REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "📊 Full Audit Report":
    st.header("📊 Full Statement Analysis")

    # Show cached report if already generated
    if "report" in st.session_state:
        st.info("✅ Report already generated. Click below to regenerate with fresh data.")

    if st.button("🔍 Generate Full Audit Report", type="primary"):
        try:
            with st.spinner("Extracting every detail… this may take 30–60 seconds."):
                report = get_detailed_report(
                    st.session_state.opening_balance,
                    st.session_state.closing_balance,
                    st.session_state.first_page_text,
                    st.session_state.raw_docs,
                )
                st.session_state.report = report

        except Exception as e:
            st.error(f"Error generating report: {e}")
            st.exception(e)
            st.stop()

    # Render report if available (persists across reruns)
    if "report" in st.session_state:
        report = st.session_state.report

        # ── Account Info ──
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

        # ── Key Metrics ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Opening Balance",   f"₹{st.session_state.opening_balance:,.2f}")
        m2.metric("Total Debits",      f"₹{report.total_debits:,.2f}")
        m3.metric("Total Credits",     f"₹{report.total_credits:,.2f}")
        m4.metric("Closing Balance",   f"₹{st.session_state.closing_balance:,.2f}")

        st.divider()

        # ── Transaction Table ──
        st.subheader("📑 Transaction History")
        df = pd.DataFrame([t.model_dump() for t in report.transactions])
        st.dataframe(df, use_container_width=True, height=300)  # FIX: was missing

        # ── Spending Chart ──
        st.subheader("📊 Spending Breakdown")
        spending_df = df[df["debit"] > 0].copy()  # FIX: defined here, not inside try block

        if not spending_df.empty:
            chart_data = spending_df.groupby("category")["debit"].sum().reset_index()
            fig = px.pie(
                chart_data, values="debit", names="category",
                hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=None,  # adapts to dark/light
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── PDF Download ──
            st.success("✅ Analysis complete!")
            pdf_buffer = generate_pdf_report(report, fig)
            st.download_button(
                label="📥 Download Full PDF Report",
                data=pdf_buffer,
                file_name=f"FinanceReport_{report.account_info.customer_name}.pdf",
                mime="application/pdf",
            )
        else:
            st.info("No debit transactions found — all transactions are incoming credits.")

        st.divider()

        # ── Security Alerts ──
        st.subheader("🚩 AI Security Alerts")
        with st.spinner("Scanning for anomalies…"):
            anomalies = get_finance_advice(
                "List all suspicious, unusually large, or duplicate transactions "
                "as bullet points with dates and amounts.",
                st.session_state.db,
            )
        st.warning(anomalies)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BUDGET PLANNER
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "🎯 Budget Planner":
    st.header("🎯 Smart Budget Planner")

    if "report" not in st.session_state:
        st.warning("⚠️ Please generate the Full Audit Report first (Page 2).")
        st.stop()

    report = st.session_state.report
    df     = pd.DataFrame([t.model_dump() for t in report.transactions])
    df["date_dt"]    = pd.to_datetime(df["txn_date"], format="%d-%m-%Y", errors="coerce")
    df["month_year"] = df["date_dt"].dt.strftime("%B %Y")

    available_months = df["month_year"].dropna().unique().tolist()
    selected_month   = st.selectbox("📅 Select Month to Analyse", available_months)

    with st.form("budget_form"):
        st.subheader(f"Set Your Goals for {selected_month}")
        user_goals = {}
        cols = st.columns(2)
        for i, cat in enumerate(st.session_state.budgets.keys()):
            user_goals[cat] = cols[i % 2].number_input(
                f"{cat} (₹)", value=st.session_state.budgets[cat], step=500
            )
        submitted = st.form_submit_button("🔥 Check My Budget", type="primary")

    if submitted:
        st.divider()
        st.subheader(f"📊 Budget vs Actual — {selected_month}")
        month_df = df[df["month_year"] == selected_month]

        for cat, goal in user_goals.items():
            actual  = float(month_df[month_df["category"] == cat]["debit"].sum())
            diff    = goal - actual
            percent = min(actual / goal, 1.0) if goal > 0 else 0.0

            with st.container():
                col_l, col_r = st.columns([3, 1])
                with col_l:
                    st.markdown(f"**{cat}**")
                    st.caption(f"Spent ₹{actual:,.0f}  ·  Goal ₹{goal:,.0f}")
                    # Color the bar based on status
                    bar_color = "normal" if diff >= 0 else "inverse"
                    st.progress(percent)
                with col_r:
                    if diff >= 0:
                        st.success(f"✅ Under\n₹{diff:,.0f}")
                    else:
                        st.error(f"🚨 Over\n₹{abs(diff):,.0f}")
            st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SPENDING FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "🔮 Spending Forecast":
    st.header("🔮 AI Spending Forecast")

    if "report" not in st.session_state:
        st.info("👈 Generate the Full Audit Report first to unlock forecasting.")
        st.stop()

    data_pack, error = calculate_forecast(st.session_state.report.transactions)

    if error:
        st.warning(error)
        st.stop()

    monthly_df, next_date, pred_val = data_pack

    # ── Chart ──
    future_df = pd.DataFrame({
        "date_dt": [monthly_df["date_dt"].iloc[-1], next_date],
        "debit":   [monthly_df["debit"].iloc[-1],   pred_val],
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_df["date_dt"], y=monthly_df["debit"],
        name="Actual Spend",
        line=dict(color="#6366f1", width=3),
        mode="lines+markers",
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=future_df["date_dt"], y=future_df["debit"],
        name="Forecast",
        line=dict(color="#a855f7", width=3, dash="dot"),
        mode="lines+markers",
        marker=dict(size=10, symbol="star"),
    ))
    fig.update_layout(
        title="Monthly Spending Trend & Forecast",
        xaxis_title="Month",
        yaxis_title="Total Spend (₹)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(fig, use_container_width=True)

    # ── AI Insights ──
    with st.spinner("AI analysing your spending trends…"):
        insights = get_forecast_insights(monthly_df, pred_val)

    last_actual = monthly_df["debit"].iloc[-1]
    delta       = pred_val - last_actual

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "Predicted Next Month",
            f"₹{pred_val:,.2f}",
            delta=f"{'▲' if delta > 0 else '▼'} ₹{abs(delta):,.0f} vs last month",
            delta_color="inverse",
        )
        st.info(f"**📈 Trend Analysis**\n\n{insights.trend_analysis}")
    with c2:
        st.warning(
            "**⚠️ Risk Alerts**\n\n"
            + "\n".join(f"- {r}" for r in insights.risk_warnings)
        )
        st.success(
            "**💡 Saving Tips**\n\n"
            + "\n".join(f"- {t}" for t in insights.saving_tips)
        )