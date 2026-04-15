from transformers import pipeline
from retriever import search_query

qa_pipeline = pipeline("text-generation", model="google/flan-t5-base")

def generate_answer(query):
    docs = search_query(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a helpful AI assistant.

    Answer ONLY from the given context.
    If answer not found, say "Not in document".

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = qa_pipeline(prompt, max_length=200, do_sample=False)

    return response[0]['generated_text']

if __name__ == "__main__":
    query = input("Ask your question: ")
    answer = generate_answer(query)

    print("\n🤖 AI Answer:\n")
    print(answer)