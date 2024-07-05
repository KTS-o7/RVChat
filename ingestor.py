import os
import warnings
import asyncio
from tqdm import tqdm

from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import (PyPDFLoader, DirectoryLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

warnings.simplefilter(action='ignore')

async def process_document(doc):
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunkedDocuments = textSplitter.split_documents([doc])
    content = [chunk.page_content for chunk in chunkedDocuments]
    metadatas = [chunk.metadata for chunk in chunkedDocuments]
    return content, metadatas

async def create_vector_database():
    pdfDirecLoader = DirectoryLoader("./files/", glob="*.pdf", loader_cls=PyPDFLoader)
    loadedDocuments = pdfDirecLoader.load()
    print(f"Loaded {len(loadedDocuments)} documents.")

    # Process documents in parallel
    results = await asyncio.gather(*[process_document(doc) for doc in loadedDocuments])
    
    # Flatten results
    content = [item for sublist in results for item in sublist[0]]
    metadatas = [item for sublist in results for item in sublist[1]]
    
    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True, num_gpu=1)
    
    vectorDB = Chroma.from_texts(texts=content, embedding=ollama_embeddings, metadatas=metadatas, persist_directory="./data")
    vectorDB.persist()

if __name__ == "__main__":
    asyncio.run(create_vector_database())
