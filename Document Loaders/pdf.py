from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("./Document Loaders/Black_book_F.pdf")

docs = data.load()

print(docs[1])