from embeddings import create_vector_store

def search_query(query):
    # Step 1: Load vector DB
    vector_db = create_vector_store("data/sample.pdf")

    # Step 2: Search similar chunks
    # This will return the top 3 most similar chunks from the vector DB based on the query
    results = vector_db.similarity_search(query, k=3) # Adjust k for more or fewer results

    # Step 3: Return results
    
    return results

if __name__ == "__main__":
    query = input("Enter your question: ")

    results = search_query(query)

    print("\nTop Results:\n")

    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:300])