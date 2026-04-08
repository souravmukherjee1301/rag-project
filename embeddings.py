from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from text_splitter import split_text

def create_vector_store(file_path):
    chunks = split_text(file_path)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

if __name__ == "__main__":
    vector_db = create_vector_store("data/sample.pdf")

    print("Vector DB created successfully!")
    print("Total vectors:", vector_db.index.ntotal)