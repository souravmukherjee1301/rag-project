from transformers import pipeline
from retriever import search_query

qa_pipeline = pipeline("text-generation", model="google/flan-t5-base")

def generate_answer(query):
    docs = search_query(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an intelligent AI assistant.

    Use ONLY the information from the provided context.
    Do NOT make up answers.
    If the answer is not in the context, say: "Not found in document".

    Give a clear and structured answer.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = qa_pipeline(prompt, max_length=200, do_sample=False)

    answer = response[0]['generated_text']

    # Clean output
    answer = answer.split("Answer:")[-1].strip()

    return answer

if __name__ == "__main__":
    query = input("Ask your question: ")

    # Step 1: Get docs (important)
    docs = search_query(query)

    # Step 2: Generate answer
    answer = generate_answer(query)

    print("\n🤖 AI Answer:\n")
    print(answer)

    # Step 3: Show sources
    print("\n📚 Sources Used:")
    for i, doc in enumerate(docs):
        print(f"\n📄 Source {i+1}:")
        print(doc.page_content[:200])