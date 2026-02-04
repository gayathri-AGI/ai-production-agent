from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rag_loader import load_knowledge_base


def build_vector_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = load_knowledge_base()

    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local("rag_vector_db")

    print("Vector database created and saved as 'rag_vector_db'")


if __name__ == "__main__":
    build_vector_database()
