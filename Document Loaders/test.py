from langchain_community.document_loaders import TextLoader

Data = TextLoader("notes.txt")

docs = Data.load()
print(docs[0])