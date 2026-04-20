from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=2,   # number of documents to load
    load_all_available_meta=True
)

docs = retriever.invoke("quantum computing")

for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print("Title:", doc.metadata.get('title'))
    print("Authors:", doc.metadata.get('authors'))
    print("Summary:", doc.page_content[:500])  # Print the first 500 characters of the summary
    print("\n")
