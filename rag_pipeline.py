from transformers import pipeline
from retriever import search_query

# Load free LLM (small model)
qa_pipeline = pipeline("text-generation", model="google/flan-t5-base")

def generate_answer(query):
    # Step 1: Get relevant chunks
    docs = search_query(query)

    # Step 2: Combine context
    context = "\n".join([doc.page_content for doc in docs])

    # Step 3: Create prompt
    prompt = f"""
    Answer the question based on the context below.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # Step 4: Generate answer
    response = qa_pipeline(prompt, max_length=200, do_sample=False)

    return response[0]['generated_text']

if __name__ == "__main__":
    query = input("Ask your question: ")

    answer = generate_answer(query)

    print("\n🤖 AI Answer:\n")
    print(answer)