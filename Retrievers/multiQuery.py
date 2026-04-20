from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI , MistralAIEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used to minimize a function by iteratively moving towards the steepest descent direction."),
    Document(page_content=" Gradient decent It is commonly used in machine learning and deep learning to optimize model parameters and minimize the loss function."),
    Document(page_content="Gradient decent is a optimization that miinimize the loss function"),
    Document(page_content="NLP stands for Natural Language Processing, which is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. It involves the development of algorithms and models that enable computers to understand, interpret, and generate human language in a meaningful way.")
]


embedding = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(docs, embedding)

retriever = vectorstore.as_retriever()

llm = ChatMistralAI(model="mistral-small-2506")

Multi_Query_Retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)

query = "What is gradient descent  ?"

res = Multi_Query_Retriever.invoke(query)

print("\nRetriever Result : \n")

for doc in res:
    print(doc.page_content)