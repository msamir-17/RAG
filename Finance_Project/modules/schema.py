from pydantic import BaseModel, Field
from typing import List, Optional

# BUG FIX: Added 'category' field — was missing but app.py checks for it
CATEGORY_OPTIONS = (
    "Food & Dining, Shopping, Travel & Transport, Entertainment, "
    "Utilities & Bills, Healthcare, Education, UPI Transfer, "
    "Cash Withdrawal, Salary / Income, Investment, Other"
)

class Transaction(BaseModel):
    sl_no: int = Field(description="The Serial Number (S.No) of the transaction")
    txn_date: str = Field(description="Transaction Date")
    value_date: str = Field(description="Value Date")
    description: str = Field(description="Complete description of the transaction")
    cheque_no: Optional[str] = Field(None, description="Cheque or Reference Number if present")
    debit: float = Field(default=0.0, description="Amount debited (money going OUT). Use 0 if not a debit.")
    credit: float = Field(default=0.0, description="Amount credited (money coming IN). Use 0 if not a credit.")
    balance: float = Field(description="Running balance after this transaction")
    category: str = Field(
        description=(
            f"Assign ONE category from this list only: {CATEGORY_OPTIONS}. "
            "Base it on the transaction description."
        )
    )

class AccountDetails(BaseModel):
    customer_name: str = Field(description="Full name of the account holder as printed on the statement")
    account_number: str = Field(description="Account number as printed on the statement")
    account_type: str = Field(description="Type of account e.g. SBA, Current")
    ifsc_code: str = Field(description="IFSC code of the branch")
    micr_code: str = Field(description="MICR code of the branch")
    branch_name: str = Field(description="Name of the bank branch")
    customer_address: str = Field(description="Customer address as printed")
    branch_address: str = Field(description="Branch address as printed")
    statement_period: str = Field(description="Date range of the statement e.g. 01-06-2022 to 12-12-2022")

class FullStatementReport(BaseModel):
    account_info: AccountDetails
    transactions: List[Transaction]
    total_debits: float = Field(description="Sum of ALL values in the Debit column only")
    total_credits: float = Field(description="Sum of ALL values in the Credit column only")
    opening_balance: float = Field(description="Balance shown on the very FIRST row (S.No 1)")
    closing_balance: float = Field(description="Balance shown on the very LAST row of the statement")


class ForecastInsight(BaseModel):
    trend_analysis: str = Field(description="Briefly explain if spending is going up or down and why")
    risk_warnings: List[str] = Field(description="List potential risks (e.g., 'Shopping is growing too fast')")
    saving_tips: List[str] = Field(description="3 actionable tips to reduce the predicted spend")