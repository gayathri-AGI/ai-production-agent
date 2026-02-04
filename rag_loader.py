import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_knowledge_base():
    documents = []

    knowledge_path = "knowledge_base"

    for file_name in os.listdir(knowledge_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(knowledge_path, file_name)
            loader = TextLoader(file_path)
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_knowledge_base()
    print(f"Loaded {len(docs)} document chunks from knowledge base.")
