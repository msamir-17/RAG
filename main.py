from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv() 

Data = TextLoader("./Document Loaders/Black_book_F.pdf")

docs = Data.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

template = ChatPromptTemplate.from_messages(
    [("system","You Are a ai that summarizes the text"),
     ("human","{Data}")]
)


model = ChatMistralAI(model="mistral-small-2506")

prompt = template.format_messages(Data = docs[0].page_content)

res = model.invoke(prompt)

print(res.content)

# 