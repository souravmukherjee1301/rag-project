from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

if __name__ == "__main__":
    file_path = "data/sample.pdf"  # Replace with your PDF file path
    documents = load_pdf(file_path)
    print(f"Loaded {len(documents)} pages from {file_path}")
    print("\nContent of the documents:\n")
    for doc in documents:
        print(doc.page_content[:100])  # Print the first 100 characters of each page's content