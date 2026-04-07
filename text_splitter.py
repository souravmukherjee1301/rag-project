from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_pdf

def split_text(file_path):
    documents = load_pdf(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    chunks = split_text("data/sample.pdf")
    
    print("Total chunks:", len(chunks))
    print("\nFirst chunk:\n")
    print(chunks[0].page_content)
    print("\nSecond chunk:\n")
    print(chunks[1].page_content)