import os
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

ollama_embeddings = OllamaEmbeddings(model='mistral')

class RVChat:
    def __init__(self):
        self.vectorDB = Chroma(persist_directory="./data",embedding_function=ollama_embeddings)
        self.ollama = ChatOllama(model='mistral')
        self.embeddings = OllamaEmbeddings(model='mistral')
        
    def nearestSearch(self, text):
        #query = OllamaEmbeddings(text)
        results = self.vectorDB.similarity_search(text,k=3)
        if results:
            return results
        else:
            return "No similar documents found."
    def nearestSearch2(self,text):
        results = self.vectorDB.max_marginal_relevance_search(text,k=3)
        return results
        

def main():
    chat = RVChat()
    #res = chat.nearestSearch("What is network ?")
    otr = chat.nearestSearch2("What is network ?")
    #for obj in res:
     #   print(obj.page_content,obj.metadata)
    for obj in otr:
        print(obj.page_content,obj.metadata)
    
main()
