""" get the modules and packages necessary """
import PyPDF2
from langchain.docstore.document import Document
import os
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings


""" create the textbook list """
book_paths = [
    "textbooks/Davidson's Principles and Practice of Medicine.pdf",
    "textbooks/Harrison's 10 Emergency Medicine.pdf",
    "textbooks/Harrison's Extra Edge Kai.pdf",
    "textbooks/Marino's The ICU Book.pdf",
    "textbooks/The Washington Manual of Critical Care.pdf"
]

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

text = ""
for book_path in book_paths:
    print(f'{book_path} starting now!')
    extracted_text = extract_text_from_pdf(book_path)
    text += extracted_text + "\n\n"
    print(f'{book_path} done extracting!')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_text(text)

documents = [Document(page_content=chunk) for chunk in chunks]

persist_directory = "chroma_db_5_tb"
os.makedirs(persist_directory, exist_ok=True)

vector_db = Chroma.from_documents(
    documents=documents,
    embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    persist_directory=persist_directory,
    collection_name="local-rag"
)

print(f"Number of documents in Chroma: {vector_db._collection.count()}")