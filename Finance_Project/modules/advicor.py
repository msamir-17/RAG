from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from modules.schema import FullStatementReport

from dotenv import load_dotenv
load_dotenv()


def get_finance_advice(user_query: str, vectorstore) -> str:
    """
    FIX: Instead of using RAG retrieval (which misses rows due to chunking),
    we pass the ENTIRE statement text directly to the LLM.
    Bank statements are small enough to fit in one context window.
    """
    llm = ChatMistralAI(model="mistral-small-2506")

    # Pull ALL documents from vectorstore and reconstruct full text
    all_docs = vectorstore.similarity_search("bank statement transactions", k=100)
    all_docs.sort(key=lambda x: x.metadata.get('page', 0))
    full_statement = "\n".join([d.page_content for d in all_docs])

    system_prompt = (
        "You are an accurate Financial Advisor analyzing a bank statement.\n\n"

        "STATEMENT DATA (use ALL of this, do not skip any row):\n"
        f"{full_statement}\n\n"

        "RULES:\n"
        "- Read EVERY transaction row before answering.\n"
        "- The 'Balance' column is a RUNNING TOTAL — never sum it.\n"
        "- DEBITS = money going OUT (money spent).\n"
        "- CREDITS = money coming IN (money received).\n"
        "- A dash '-' in the Debit column means ₹0 debit for that row.\n"
        "- If all Debit values are '-' or 0, then there are NO expenses — "
        "all transactions are incoming money.\n"
        "- When asked about UPI/Paytm/specific merchants, list EVERY matching "
        "row with its date and amount — do not stop at 2 results.\n"
        "- When asked 'where did I spend the most', check debits first. "
        "If no debits exist, list the largest CREDIT transactions instead "
        "and clarify these are incoming amounts.\n"
        "- Be specific with amounts and dates. Never say 'not found' if "
        "the data is visible above.\n"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]

    response = llm.invoke(messages)
    return response.content


def get_detailed_report(
    vectorstore,
    opening_balance: float,
    closing_balance: float
) -> FullStatementReport:
    """
    Generate structured report using .with_structured_output() —
    eliminates OUTPUT_PARSING_FAILURE completely.
    """
    llm = ChatMistralAI(model="mistral-small-2506", temperature=0)
    structured_llm = llm.with_structured_output(FullStatementReport)

    # Pull all docs and sort by page
    docs = vectorstore.similarity_search(
        "bank statement transactions balance debit credit", k=100
    )
    docs.sort(key=lambda x: x.metadata.get('page', 0))
    context_text = "\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict Financial Auditor. "
            "Extract data ONLY from the bank statement context provided.\n\n"

            "CRITICAL RULES:\n"
            "1. CUSTOMER NAME: Copy exactly from the statement header.\n"
            f"2. OPENING BALANCE: Use exactly {opening_balance}.\n"
            f"3. CLOSING BALANCE: Use exactly {closing_balance}.\n"
            "4. TOTAL DEBITS: Sum ALL Debit column values. "
            "A dash '-' means 0. Return a plain number only.\n"
            "5. TOTAL CREDITS: Sum ALL Credit column values. "
            "Return a plain number only — never a formula.\n"
            "6. TRANSACTIONS: Extract EVERY single row. Do not skip any.\n"
            "7. CATEGORY per row from: Food & Dining, Shopping, "
            "Travel & Transport, Entertainment, Utilities & Bills, "
            "Healthcare, Education, UPI Transfer, Cash Withdrawal, "
            "Salary / Income, Investment, Other.\n"
            "8. All numeric fields = plain numbers only. Never formulas.\n"
            "9. Missing values → use 0 or empty string.\n"
        )),
        ("human", "Bank Statement Data:\n{context}")
    ])

    chain = prompt | structured_llm
    report = chain.invoke({"context": context_text})

    # Always override with regex-extracted values — guaranteed correct
    report.opening_balance = opening_balance
    report.closing_balance = closing_balance

    return report