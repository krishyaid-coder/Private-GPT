from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS

presist_directory = 'db'

def main():
    for root, dirs, files in os.walk('docs'):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 500)
    # chunking is done here
    texts = text_splitter.split_documents(documents)
    # creating embeddings for this chunks
    embeddings = SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2')
    # create the vector_store and store these embeddings inside it
    db = Chroma.from_documents(texts, embeddings, persist_directory = presist_directory)
    db.persist()
    db = None 

if __name__ == "__main__":
    main()