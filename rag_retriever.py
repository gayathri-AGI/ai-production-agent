"""
RAG Retriever Module
Agentic Security-Focused RAG
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def retrieve_security_context(query: str) -> str:
    """
    Retrieves relevant security rules from the FAISS vector database
    to ground LLM responses and prevent unsafe behavior.
    """

    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS vector database (trusted local source)
    db = FAISS.load_local(
        "rag_vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve top relevant chunks
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)

    # Combine retrieved content
    context = "\n".join(doc.page_content for doc in docs)

    # Fallback if no relevant docs found
    if not context.strip():
        context = (
            "No specific security rules were retrieved. "
            "Follow general safety and ethical guidelines."
        )

    return context
