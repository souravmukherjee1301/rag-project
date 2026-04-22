from embeddings import load_vector_db
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def search_query(query):
    # Step 1: Load vector DB
    vector_db = load_vector_db() # Ensure this matches the path where the vector store was saved during creation

    # Step 2: Search similar chunks
    # This will return the top 5 most similar chunks from the vector DB based on the query
    results = vector_db.similarity_search(query, k=5) # Adjust k for more or fewer results

    print(f"DEBUG: Found {len(results)} similar chunks for the query.")

    # Step 3: Return results
    
    return results



# if __name__ == "__main__":
#     query = input("Enter your question: ")

#     results = search_query(query)

#     print("\nTop Results:\n")

#     for i, doc in enumerate(results):
#         print(f"\nResult {i+1}:")
#         print(doc.page_content[:300])