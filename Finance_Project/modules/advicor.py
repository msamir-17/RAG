from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
from langchain_classic.chains import create_retrieval_chain


from dotenv import load_dotenv

load_dotenv()

def get_finance_advice(user_query, storage_directory="storage/chroma_db"):
    """
    1. Connects to the saved ChromaDB
    2. Sets up a 'Finance Advisor' prompt
    3. Retrieves relevant chunks and generates an answer
    """
    
    # 1. Load the existing Vector DB
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = Chroma(
        persist_directory=storage_directory, 
        embedding_function=embeddings
    )
    
    # 2. Create the Retriever 
    # (Using search_kwargs={"k": 3} to get top 3 most relevant transaction chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 3. Define the LLM
    llm = ChatMistralAI(model="mistral-small-2506")
    
    # 4. Create a specialized Finance Prompt
    system_prompt = (
        "You are a professional Financial Advisor. Use the following pieces of "
        "retrieved bank statements to answer the user's question. "
        "If you don't know the answer based on the data, just say you don't know. "
        "Provide insights on spending habits if asked."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 5. Create the RAG Chain
    # 'create_stuff_documents_chain' handles passing context to the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # 'create_retrieval_chain' connects the retriever to the LLM chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # 6. Invoke the chain
    response = rag_chain.invoke({"input": user_query})
    
    return response["answer"]