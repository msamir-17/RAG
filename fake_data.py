# from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
# from reportlab.lib import colors
# import random
# from datetime import datetime, timedelta

# doc = SimpleDocTemplate("statement.pdf")

# # Table header
# data = [["Date", "Description", "Amount (₹)", "Type"]]

# categories = ["Swiggy", "Zomato", "Uber", "Amazon", "Flipkart", "Electricity", "Netflix", "Petrol", "Grocery", "Gym Membership", "Mobile Recharge", "Dining Out", "Movie Tickets", "Clothing", "Travel", "Health Insurance", "Education", "Subscription", "Miscellaneous", "Rent", "Loan Payment", "Credit Card Bill", "Investment", "Donation", "Gift", "Entertainment", "Software Subscription", "Hardware Purchase", "Professional Services", "Personal Care", "Automobile Maintenance", "Home Improvement", "Furniture", "Appliances", "Books", "Stationery", "Office Supplies", "Childcare", "E-commerce", "Financial Services", "Legal Services", "Consulting Services", "Medical Expenses", "Pharmacy", "Veterinary Services", "Fitness Classes", "Sports Equipment", "Outdoor Activities", "Hobbies and Crafts", "Music and Arts", "Charity Contributions"]

# start_date = datetime(2026, 3, 1)

# # Generate 1000 rows
# for i in range(1000):
#     date = start_date + timedelta(days=random.randint(0, 30))
#     desc = random.choice(categories)
#     amount = random.randint(100, 5000)
    
#     data.append([
#         date.strftime("%d-%m-%Y"),
#         desc,
#         f"-{amount}",
#         "Debit"
#     ])

# table = Table(data)

# style = TableStyle([
#     ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
#     ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
#     ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
# ])

# table.setStyle(style)

# doc.build([table])

# print("PDF created: statement.pdf")

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime, timedelta
import random

doc = SimpleDocTemplate("statement.pdf")
styles = getSampleStyleSheet()

elements = []

# 🔴 Header
title = Paragraph("<b>Account Statement from 01-06-2022 to 12-12-2022</b>", styles['Title'])
elements.append(title)
elements.append(Spacer(1, 20))

# 🧾 Account Details
details = [
    ["Customer Name:", "Samir Sharma", "Branch Name:", "Mumbai"],
    ["Account Number:", "1234567890", "IFSC Code:", "SBIN0001234"],
    ["Account Type:", "Savings", "MICR Code:", "400002123"],
    ["Customer Address:", "Virar, Maharashtra, India", "Branch Address:", "Mumbai Main Branch"]
]

table_details = Table(details, colWidths=[120, 180, 120, 180])

table_details.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), colors.whitesmoke),
    ("GRID", (0,0), (-1,-1), 0.5, colors.black)
]))

elements.append(table_details)
elements.append(Spacer(1, 20))

# 📊 Transaction Table Header
data = [[
    "S.No", "Transaction Date", "Value Date", "Description",
    "Debit", "Credit", "Balance"
]]

balance = 50000

categories = ["Swiggy", "Zomato", "Uber", "Amazon", "Flipkart", "Electricity"]

start_date = datetime(2022, 6, 1)

# 🔁 Generate Transactions
for i in range(1, 101):  # you can increase to 1000+
    date = start_date + timedelta(days=random.randint(0, 180))
    desc = random.choice(categories)

    if random.choice([True, False]):
        debit = random.randint(100, 5000)
        credit = ""
        balance -= debit
    else:
        credit = random.randint(500, 10000)
        debit = ""
        balance += credit

    data.append([
        i,
        date.strftime("%d-%m-%Y"),
        date.strftime("%d-%m-%Y"),
        desc,
        debit,
        credit,
        balance
    ])

# 📊 Table
table = Table(data, repeatRows=1)

table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.grey),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("GRID", (0,0), (-1,-1), 0.25, colors.black),
    ("FONTSIZE", (0,0), (-1,-1), 8)
]))

elements.append(table)

# 📄 Build PDF
doc.build(elements)

print("✅ Professional statement.pdf created")