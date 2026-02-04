from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def retrieve_security_context(query: str) -> str:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local("rag_vector_db", embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(query)

    return "\n".join(doc.page_content for doc in docs)
