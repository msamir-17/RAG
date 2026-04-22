from pydantic import BaseModel, Field
from typing import List, Optional

class Transaction(BaseModel):
    sl_no: int = Field(description="The Serial Number of the transaction")
    txn_date: str = Field(description="Transaction Date")
    value_date: str = Field(description="Value Date")
    description: str = Field(description="Complete description of the transaction")
    cheque_no: Optional[str] = Field(None, description="Cheque or Reference Number")
    debit: float = Field(default=0.0, description="Amount debited (money out)")
    credit: float = Field(default=0.0, description="Amount credited (money in)")
    balance: float = Field(description="Closing balance after this transaction")

class AccountDetails(BaseModel):
    customer_name: str
    account_number: str
    account_type: str
    ifsc_code: str
    micr_code: str
    branch_name: str
    customer_address: str
    branch_address: str
    statement_period: str = Field(description="The date range of the statement")

class FullStatementReport(BaseModel):
    account_info: AccountDetails
    transactions: List[Transaction]
    total_debits: float
    total_credits: float
    opening_balance: float
    closing_balance: float

class StatementSummary(BaseModel):
    account_holder: str
    opening_balance: float = Field(description="Balance BEFORE the first listed transaction (S.No 1)")
    closing_balance: float = Field(description="Balance AFTER the very last listed transaction (S.No 100)")
    total_debit: float = Field(description="Sum of all amounts in the Debit column")
    total_credit: float = Field(description="Sum of all amounts in the Credit column")
    transactions: List[Transaction]