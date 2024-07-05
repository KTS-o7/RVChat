import os
import asyncio
import chainlit as cl
import warnings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.chat_models.ollama import ChatOllama
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

llmmodel = os.getenv("LLM_MODEL", "llama3")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

async def process_documents(directory):
    pdfDirecLoader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    loadedDocuments = pdfDirecLoader.load()
    chunkedDocuments = text_splitter.split_documents(loadedDocuments)
    content = [doc.page_content for doc in chunkedDocuments]
    metadatas = [doc.metadata for doc in chunkedDocuments]
    return content, metadatas

@cl.on_chat_start
async def on_chat_start():
    warnings.simplefilter(action='ignore')
    fast_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorDB = Chroma(persist_directory="./data", embedding_function=fast_embeddings)
    
    msg = cl.Message(content="Processing Started ...")
    await msg.send()
    
    if not os.path.exists("./files"):
        os.makedirs("./files")
    if not os.path.exists("./data"):
        os.makedirs("./data")
        content, metadatas = await process_documents("./files/")
        vectorDB = Chroma.from_texts(texts=content, embedding=fast_embeddings, metadatas=metadatas, persist_directory="./data")
        vectorDB.persist()

    vectorDB = Chroma(persist_directory="./data", embedding_function=fast_embeddings)
    
    messageHistory = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=messageHistory,
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model=llmmodel, temperature=0.3),
        chain_type="stuff",
        retriever=vectorDB.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    
    msg.content = "Processing Complete..."
    await msg.update()
    
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(cl.Text(content=source_doc.page_content, name=source_name))
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
