import streamlit as st
import os

from embeddings import create_and_save_db
from rag_pipeline import generate_answer

st.set_page_config(page_title="RAG App", layout="centered")

st.title("AI Document Q&A System")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded file
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")

    # Create vector DB
    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            create_and_save_db(file_path)

        st.success("Document processed!")

    # Ask Question
    query = st.text_input("Ask a question:")

    if st.button("Get Answer"):
        if query:
            with st.spinner("Thinking..."):
                answer, docs = generate_answer(query)

            st.subheader("Answer:")
            st.write(answer)

            st.subheader("Sources:")
            for i, doc in enumerate(docs):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content[:200])
        else:
            st.warning("Please enter a question")