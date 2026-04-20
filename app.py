import streamlit as st
from rag_pipeline import generate_answer

st.set_page_config(page_title="RAG App", layout="centered")

st.title("AI Document Q&A System")

st.write("Ask questions from your document")

# Input box
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query:
        answer, docs = generate_answer(query)

        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Sources:")
        for i, doc in enumerate(docs):
            st.write(f"**Source {i+1}:**")
            st.write(doc.page_content[:200])
    else:
        st.warning("Please enter a question")