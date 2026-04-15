from embeddings import create_vector_store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_vector_db():
    # Load the vector store from the local file system
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    ) # Ensure this matches the model used during creation

    # Allowing dangerous deserialization for loading the FAISS index (use with caution)
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    return vector_db

def search_query(query):
    # Step 1: Load vector DB
    vector_db = load_vector_db() # Ensure this matches the path where the vector store was saved during creation

    # Step 2: Search similar chunks
    # This will return the top 3 most similar chunks from the vector DB based on the query
    results = vector_db.similarity_search(query, k=3) # Adjust k for more or fewer results

    # Step 3: Return results
    
    return results



# if __name__ == "__main__":
#     query = input("Enter your question: ")

#     results = search_query(query)

#     print("\nTop Results:\n")

#     for i, doc in enumerate(results):
#         print(f"\nResult {i+1}:")
#         print(doc.page_content[:300])