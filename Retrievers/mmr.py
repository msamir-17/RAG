from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used to minimize a function by iteratively moving towards the steepest descent direction, "),
    Document(page_content="which is determined by the negative of the gradient. "),
    Document(page_content="It is commonly used in machine learning and deep learning to optimize model parameters and minimize the loss function."),
    Document(page_content="Gradient decent is a optimization that miinimize the loss function"),
    Document(page_content="Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates the data points of different classes in a high-dimensional space. The hyperplane is chosen to maximize the margin between the classes, which helps improve the generalization of the model. SVM can also be used for regression by finding a hyperplane that best fits the data points while allowing for some margin of error.")
]

embedding = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(docs, embedding)

print("similarity retriever")

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

similarity_docs = similarity_retriever.invoke("What is gradient descent")


for doc in similarity_docs:
    print(doc.page_content)


print("===== mmr retriever =====")


mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, "fetch_k":5}
)

mmr_docs = mmr_retriever.invoke("What is gradient descent")

for doc in mmr_docs:
    print(doc.page_content)