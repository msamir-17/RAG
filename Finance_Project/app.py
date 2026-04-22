import streamlit as st
import os
import plotly.express as px
import pandas as pd
from modules.processor import process_pdf_to_memory
from modules.advicor import get_finance_advice, get_detailed_report
from streamlit.runtime.scriptrunner import get_script_run_ctx

st.set_page_config(page_title="AI Finance Advisor", layout="wide")

# Session State
if "ready" not in st.session_state:
    st.session_state.ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "opening_balance" not in st.session_state:
    st.session_state.opening_balance = 0.0
if "closing_balance" not in st.session_state:
    st.session_state.closing_balance = 0.0

ctx = get_script_run_ctx()
session_id = ctx.session_id

st.title("💰 AI Personal Finance Advisor")
st.markdown("Upload your bank statement and ask questions about your spending.")

# Sidebar
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Choose a bank statement (PDF)", type="pdf")

    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", "temp_statement.pdf")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing your statement..."):
            # processor now returns 3 values
            db, opening, closing = process_pdf_to_memory(file_path)
            st.session_state.db = db
            st.session_state.opening_balance = opening
            st.session_state.closing_balance = closing
            st.session_state.ready = True
            st.success("Statement processed!")

        # Show extracted balances in sidebar for quick verification
        st.markdown("---")
        st.markdown(f"**Opening Balance:** ₹{opening:,.2f}")
        st.markdown(f"**Closing Balance:** ₹{closing:,.2f}")

if st.session_state.ready:
    tab1, tab2 = st.tabs(["💬 Chat Advisor", "📊 Full Audit Report"])

    with tab1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about your transactions..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = get_finance_advice(prompt, st.session_state.db)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

    with tab2:
        st.header("Full Statement Analysis")
        if st.button("🔍 Generate Full Audit Report"):
            try:
                with st.spinner("Extracting every detail..."):
                    # Pass pre-extracted balances — AI cannot hallucinate these now
                    report = get_detailed_report(
                        st.session_state.db,
                        st.session_state.opening_balance,
                        st.session_state.closing_balance
                    )

                    st.subheader("🏦 Account Information")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.info(f"**Customer:** {report.account_info.customer_name}\n\n"
                                f"**A/C No:** {report.account_info.account_number}\n\n"
                                f"**Account Type:** {report.account_info.account_type}")
                    with c2:
                        st.info(f"**IFSC:** {report.account_info.ifsc_code}\n\n"
                                f"**Branch:** {report.account_info.branch_name}\n\n"
                                f"**Period:** {report.account_info.statement_period}")

                    st.divider()
                    m1, m2, m3 = st.columns(3)
                    # Use session state values (from raw text) — guaranteed correct
                    m1.metric("Opening Balance", f"₹{st.session_state.opening_balance:,.2f}")
                    m2.metric("Total Debits (Out)", f"₹{report.total_debits:,.2f}", delta_color="inverse")
                    m3.metric("Closing Balance", f"₹{st.session_state.closing_balance:,.2f}")

                    st.subheader("📑 Itemized Transaction History")
                    df = pd.DataFrame([t.model_dump() for t in report.transactions])
                    st.dataframe(df, use_container_width=True)

                    st.subheader("📊 Spending Breakdown")
                    if not df.empty and 'category' in df.columns:
                        spending_df = df[df['debit'] > 0].copy()
                        if not spending_df.empty:
                            chart_data = spending_df.groupby('category')['debit'].sum().reset_index()
                            fig = px.pie(chart_data, values='debit', names='category',
                                         title='Expenses by Category', hole=0.4)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No debit transactions found to categorize.")
                    else:
                        st.error("Category data missing. Please regenerate the report.")

                    st.divider()
                    st.subheader("🚩 AI Security Alerts")
                    security_query = "List all suspicious or unusually large transactions as bullet points with amounts."
                    anomalies = get_finance_advice(security_query, st.session_state.db)
                    st.warning(anomalies)

            except Exception as e:
                st.error(f"Error generating report: {e}")
                st.exception(e)
else:
    st.info("👈 Please upload a bank statement in the sidebar to begin.")
    st.image("https://cdn-icons-png.flaticon.com/512/1611/1611179.png", width=100)