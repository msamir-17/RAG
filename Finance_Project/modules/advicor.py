from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.output_parsers import PydanticOutputParser
from modules.schema import FullStatementReport

from dotenv import load_dotenv

load_dotenv()

def get_finance_advice(user_query, vectorstore):
    # Use the memory object directly
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    llm = ChatMistralAI(model="mistral-small-2506")
    
    system_prompt = (
        "You are a concise Financial Advisor. "
        "IMPORTANT: The 'Balance' column in the text is a running total. DO NOT sum it up. "
        "To find the current balance, look ONLY at the very last row. "
        "Give short, direct answers. Do not show your step-by-step math unless asked."
        "\n\nContext: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    response = chain.invoke({"input": user_query})
    return response["answer"]

def get_detailed_report(vectorstore):
    llm = ChatMistralAI(model="mistral-small-2506", temperature=0) # Temperature 0 = No guessing
    parser = PydanticOutputParser(pydantic_object=FullStatementReport)

    # Grab the whole context
    docs = vectorstore.similarity_search("transactions", k=50)
    # SORTING IS CRITICAL: Ensure the AI reads S.No 1 first and S.No 100 last
    docs.sort(key=lambda x: x.metadata.get('page', 0))
    context_text = "\n".join([d.page_content for d in docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict Financial Auditor. USE ONLY THE DATA PROVIDED. "
            "1. OPENING BALANCE: Look at the 'Balance' column for S.No 1. That is the opening balance. "
            "2. CLOSING BALANCE: Look at the 'Balance' column for S.No 100. That is the closing balance. "
            "3. CATEGORIZATION: You MUST assign a category to every row. "
            "4. NO HALLUCINATION: Do not use numbers like 123,456. If a number is not in the text, use 0. "
            "5. ACCOUNT INFO: The name 'Samir Sharma' is at the very top. Use it."
            "\n{format_instructions}"
        )),
        ("human", "Context: {context}")
    ])
    
    chain = prompt | llm | parser
    return chain.invoke({"context": context_text, "format_instructions": parser.get_format_instructions()})