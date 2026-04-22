from transformers import pipeline
from retriever import search_query

qa_pipeline = pipeline("text-generation", model="gpt2") # Replace with your fine-tuned model if available

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

    response = qa_pipeline(
    prompt,
    max_new_tokens=220,
    max_length=512,
    do_sample=True,
    temperature=0.7,
    truncation=True,
    return_full_text=False
    )

    answer = response[0]['generated_text']

    # answer = response[0]['generated_text'].strip()

    # print("RAW OUTPUT:", response)

    if "Answer:" in answer:
        answer = answer.split("Answer:")[1].strip()
    else:
        answer = answer.strip()

    return answer, docs

if __name__ == "__main__":
    query = input("Ask your question: ")

    # Step 1: Get docs (important)
    docs = search_query(query)

    # Step 2: Generate answer
    answer = generate_answer(query)

    print("\n AI Answer:\n")
    print(answer)

    # Step 3: Show sources
    print("\n📚 Sources Used:")
    for i, doc in enumerate(docs):
        print(f"\n📄 Source {i+1}:")
        print(doc.page_content[:200])
