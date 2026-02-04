"""
RAG Retriever Module
Agentic Security-Focused RAG
Compatible with latest LangChain versions
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def retrieve_security_context(query: str) -> str:
    """
    Retrieves relevant security rules from the FAISS vector database
    to ground LLM responses and prevent unsafe behavior.
    """

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS vector DB (trusted local source)
    db = FAISS.load_local(
        "rag_vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Create retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 🔹 NEW LangChain API (FIX)
    docs = retriever.invoke(query)

    # Combine retrieved content
    context = "\n".join(doc.page_content for doc in docs)

    # Fallback safety
    if not context.strip():
        context = (
            "No specific security rules were retrieved. "
            "Follow general safety and ethical guidelines."
        )

    return context
