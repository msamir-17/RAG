from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

from dotenv import load_dotenv

load_dotenv() 

from langchain_core.documents import Document

docs = [
    Document(page_content="This is the content of JAVA document.", metadata={"source": "document1.txt"}),
    Document(page_content="This is the content of python document.", metadata={"source": "document2.txt"}),
]

embedding = MistralAIEmbeddings(model="mistral-embed")

vectorized = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="chroma_db"
)

res = vectorized.similarity_search("What is python ", k=2)

for i in res:
    print(i)

retriver = vectorized.as_retriever()

docs = retriver.invoke("What is python")

for d in docs:
    print(d.page_content)