import streamlit as st
import os

from embeddings import create_and_save_db
from rag_pipeline import generate_answer

if "messages" not in st.session_state:
    st.session_state.messages = []

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

    # Chat input
    query = st.chat_input("Ask your question")

    # query = st.text_input("Ask your question")

    if query:
        # st.write(f"Your question: {query}")
        # Save user message
        st.session_state.messages.append(
            {"role": "user", "content": query})

        with st.spinner("Thinking..."):
            answer, docs = generate_answer(query)

        # Save AI response
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})

        # Display chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
