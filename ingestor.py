import os
import warnings
from tqdm import tqdm

from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import (PyPDFLoader,CSVLoader, UnstructuredMarkdownLoader,UnstructuredPowerPointLoader,DirectoryLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

warnings.simplefilter(action='ignore')

def create_vector_database():
    
    pdfDirecLoader = DirectoryLoader("./files/", glob="*.pdf",loader_cls=PyPDFLoader)
    loadedDocuments = pdfDirecLoader.load()
    print(len(loadedDocuments))
    
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    chunkedDocuments = textSplitter.split_documents(loadedDocuments)
    print(len(chunkedDocuments))
    print(type(chunkedDocuments))
    
    ollama_embeddings = OllamaEmbeddings(model='phi',show_progress=True)
    
    vectorDB = Chroma.from_documents(documents=chunkedDocuments,embedding=ollama_embeddings,persist_directory="./data")
    
    
if __name__ == "__main__":
    create_vector_database()
