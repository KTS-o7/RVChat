import os
import asyncio
from typing import List
from langchain_community.document_loaders import PyPDFLoader

from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama

from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", verbose=True)

llm_model = os.getenv("LLM_MODEL", "llama3")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

async def process_text_file(file_path):
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        text = await f.read()
    texts = text_splitter.split_text(text)
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
    return texts, metadatas

async def process_pdf_file(file_path):
    loader = PyPDFLoader(file_path)
    texts = text_splitter.split_documents(loader.load())
    textCollection = [text.page_content for text in texts]
    metadatas = [text.metadata for text in texts]
    return textCollection, metadatas

@cl.on_chat_start
async def on_chat_start():
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a text file or PDF to begin!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()
    
    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    if file.type == "text/plain":
        texts, metadatas = await process_text_file(file.path)
    elif file.type == "application/pdf":
        texts, metadatas = await process_pdf_file(file.path)

    embeddings = OllamaEmbeddings(temperature=0.3, top_k=20, show_progress=True, model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOllama(model_name=llm_model, temperature=0.2, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    msg.content = f"Processing `{file.name}` done. You can now ask questions! We are using the {llm_model} model."
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
