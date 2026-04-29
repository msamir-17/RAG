import time
import pandas as pd
import plotly.io as pio
from io import BytesIO
from typing import List
from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import streamlit as st
from modules.schema import FullStatementReport, AccountDetails, Transaction, ForecastInsight

# Internal helper for transaction batching
class TransactionBatch(BaseModel):
    transactions: List[Transaction]

def get_finance_advice(user_query: str, vectorstore) -> str:
    llm = ChatMistralAI(model="mistral-small-2506")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    template = """You are a concise Financial Advisor. Answer based ONLY on context.
    Context: {context}
    Question: {question}
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain.invoke(user_query)

def process_transaction_batch_with_retry(batch_text, batch_llm, retries=2):
    for attempt in range(retries):
        try:
            return batch_llm.invoke(batch_text).transactions
        except:
            time.sleep(1) # Short pause before retry
    return []

def get_header_direct(first_page_text):
    llm = ChatMistralAI(model="mistral-small-2506", temperature=0)
    structured_llm = llm.with_structured_output(AccountDetails)
    prompt = f"Extract identity details. Use EXACT values. Do not guess.\n\nText: {first_page_text}"
    return structured_llm.invoke(prompt)

# @st.cache_data(show_spinner=False)
def get_detailed_report(opening_balance, closing_balance, first_page_text, raw_docs):
    account_info = get_header_direct(first_page_text)
    llm = ChatMistralAI(model="mistral-small-2506", temperature=0)
    batch_llm = llm.with_structured_output(TransactionBatch)
    
    full_text = "\n".join([d.page_content for d in raw_docs])
    # Large batches for speed (approx 30-40 rows)
    text_batches = [full_text[i:i+6000] for i in range(0, len(full_text), 6000)]
    
    all_transactions = []
    for batch in text_batches:
        all_transactions.extend(process_transaction_batch_with_retry(batch, batch_llm))

    return FullStatementReport(
        account_info=account_info,
        transactions=all_transactions,
        total_debits=sum(t.debit for t in all_transactions),
        total_credits=sum(t.credit for t in all_transactions),
        opening_balance=opening_balance,
        closing_balance=closing_balance
    )

def generate_pdf_report(report_data, plotly_fig):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title & Header
    story.append(Paragraph(f"Financial Audit: {report_data.account_info.customer_name}", styles['Title']))
    story.append(Spacer(1, 12))

    # Summary Table
    data = [["Label", "Value"]]
    data.append(["Opening Balance", f"₹{report_data.opening_balance:,.2f}"])
    data.append(["Total Debits", f"₹{report_data.total_debits:,.2f}"])
    data.append(["Closing Balance", f"₹{report_data.closing_balance:,.2f}"])
    
    t = Table(data, colWidths=[150, 250])
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)]))
    story.append(t)
    story.append(Spacer(1, 20))

    # Chart
    img_bytes = pio.to_image(plotly_fig, format="png")
    story.append(Image(BytesIO(img_bytes), width=400, height=200))

    doc.build(story)
    buffer.seek(0)
    return buffer

def calculate_forecast(transactions):
    df = pd.DataFrame([t.model_dump() for t in transactions])
    df['date_dt'] = pd.to_datetime(df['txn_date'], format='%d-%m-%Y', errors='coerce')
    monthly_spend = df[df['debit'] > 0].groupby(df['date_dt'].dt.to_period('M'))['debit'].sum().reset_index()
    monthly_spend['date_dt'] = monthly_spend['date_dt'].dt.to_timestamp()
    
    if len(monthly_spend) < 2:
        return None, "Insufficient data (need 2+ months)"

    avg_growth = monthly_spend['debit'].pct_change().mean()
    prediction = monthly_spend['debit'].iloc[-1] * (1 + (avg_growth or 0))
    next_date = monthly_spend['date_dt'].iloc[-1] + pd.DateOffset(months=1)
    
    return (monthly_spend, next_date, prediction), None

def get_forecast_insights(monthly_data, prediction):
    llm = ChatMistralAI(model="mistral-small-2506", temperature=0)
    structured_llm = llm.with_structured_output(ForecastInsight)
    prompt = f"Analyze these monthly totals:\n{monthly_data.to_string()}\nPrediction: ₹{prediction:,.2f}. Provide reasoning."


    return structured_llm.invoke(prompt)

def fast_intent(text: str):
    t = text.lower()

    if "report" in t or "audit" in t:
        return "audit"
    if "budget" in t:
        return "budget"
    if "forecast" in t or "future" in t:
        return "forecast"

    return "chat"