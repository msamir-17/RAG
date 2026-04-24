from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from modules.schema import FullStatementReport, AccountDetails, Transaction
from pydantic import BaseModel
from typing import List
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import plotly.io as pio

# Helper schema for batching transactions
class TransactionBatch(BaseModel):
    transactions: List[Transaction]

def get_finance_advice(user_query: str, vectorstore) -> str:
    # (Keeping your high-context chat logic as it works well for simple queries)
    llm = ChatMistralAI(model="mistral-small-2506")
    all_docs = vectorstore.similarity_search("bank statement transactions", k=100)
    all_docs.sort(key=lambda x: x.metadata.get('page', 0))
    full_statement = "\n".join([d.page_content for d in all_docs])

    system_prompt = (
        "You are an accurate Financial Advisor. Use this statement data:\n"
        f"{full_statement}\n\n"
        "Answer the user query concisely based only on this data."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]
    return llm.invoke(messages).content

def get_detailed_report(vectorstore, opening_balance: float, closing_balance: float) -> FullStatementReport:
    llm = ChatMistralAI(model="mistral-small-2506", temperature=0)
    
    # --- STEP 1: EXTRACT HEADER (Isolated Call) ---
    # We only look for the top part of the document
    header_llm = llm.with_structured_output(AccountDetails)
    header_docs = vectorstore.similarity_search("Customer Name, Account Number, IFSC, Branch, Address", k=3)
    header_context = "\n".join([d.page_content for d in header_docs])
    
    header_prompt = "Extract account identity details from this text. Ignore the transaction table.\n\n" + header_context
    account_info = header_llm.invoke(header_prompt)

    # --- STEP 2: EXTRACT TRANSACTIONS (Batch Call) ---
    # We process transactions in small batches to avoid JSON/Token breakage
    batch_llm = llm.with_structured_output(TransactionBatch)
    
    # Get all potential transaction chunks
    all_tx_docs = vectorstore.similarity_search("S.No, Date, Description, Debit, Credit, Balance", k=60)
    all_tx_docs.sort(key=lambda x: x.metadata.get('page', 0))
    
    all_transactions = []
    
    # Batch size of 4 chunks (usually ~15-20 rows)
    for i in range(0, len(all_tx_docs), 4):
        batch_context = "\n".join([d.page_content for d in all_tx_docs[i : i + 4]])
        batch_prompt = f"Extract every transaction row from this text into JSON. Categorize each row accurately.\n\nContext:\n{batch_context}"
        
        try:
            res = batch_llm.invoke(batch_prompt)
            all_transactions.extend(res.transactions)
        except Exception as e:
            print(f"Batch {i} failed, skipping... Error: {e}")

    # --- STEP 3: PYTHON MATH (100% Accuracy) ---
    total_debits = sum(t.debit for t in all_transactions)
    total_credits = sum(t.credit for t in all_transactions)

    # --- STEP 4: FINAL ASSEMBLY ---
    return FullStatementReport(
        account_info=account_info,
        transactions=all_transactions,
        total_debits=total_debits,
        total_credits=total_credits,
        opening_balance=opening_balance,
        closing_balance=closing_balance
    )



def generate_pdf_report(report_data, plotly_fig):
    buffer = BytesIO()
    # A4 Page with 1-inch margins
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    style_title = styles['Title']
    style_title.textColor = colors.HexColor("#1e3a8a") # Professional Dark Blue
    
    story = []

    # 1. HEADER TITLE
    story.append(Paragraph("AI FINANCIAL AUDIT REPORT", style_title))
    story.append(Paragraph(f"Period: {report_data.account_info.statement_period}", styles['Normal']))
    story.append(Spacer(1, 20))

    # 2. ACCOUNT INFO & METRICS (Side-by-Side Table)
    # Left side: Account Info | Right side: Key Math
    summary_data = [
        [Paragraph(f"<b>Customer:</b> {report_data.account_info.customer_name}", styles['Normal']), 
         Paragraph(f"<b>Opening Balance:</b> ₹{report_data.opening_balance:,.2f}", styles['Normal'])],
        [Paragraph(f"<b>A/C No:</b> {report_data.account_info.account_number}", styles['Normal']), 
         Paragraph(f"<b>Total Debits:</b> <font color='red'>₹{report_data.total_debits:,.2f}</font>", styles['Normal'])],
        [Paragraph(f"<b>IFSC:</b> {report_data.account_info.ifsc_code}", styles['Normal']), 
         Paragraph(f"<b>Closing Balance:</b> ₹{report_data.closing_balance:,.2f}", styles['Normal'])]
    ]
    
    summary_table = Table(summary_data, colWidths=[260, 260])
    summary_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor("#f8fafc")), # Light grey for math side
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 25))

    # 3. SPENDING CHART
    story.append(Paragraph("<b>Spending Distribution</b>", styles['Heading3']))
    # Pro Tip: Tighten chart margins so it fits nicely in PDF
    plotly_fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    img_bytes = pio.to_image(plotly_fig, format="png", width=700, height=350, scale=2) # Higher scale = Sharper PDF
    story.append(Image(BytesIO(img_bytes), width=450, height=225))
    story.append(Spacer(1, 25))

    # 4. TRANSACTION TABLE
    story.append(Paragraph("<b>Itemized Transaction Log</b>", styles['Heading3']))
    data = [["Date", "Description", "Debit", "Credit", "Balance"]]
    
    for tx in report_data.transactions:
        # We color the Debit red and Credit green for readability
        data.append([
            tx.txn_date, 
            Paragraph(tx.description[:45], styles['Normal']), # Wrap long descriptions
            f"₹{tx.debit:,.0f}" if tx.debit > 0 else "-",
            f"₹{tx.credit:,.0f}" if tx.credit > 0 else "-",
            f"₹{tx.balance:,.0f}"
        ])

    tx_table = Table(data, repeatRows=1, colWidths=[70, 230, 70, 70, 80])
    tx_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")), # Header Blue
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.2, colors.HexColor("#e2e8f0")),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]), # Zebra Stripes
    ]))
    story.append(tx_table)

    doc.build(story)
    buffer.seek(0)
    return buffer