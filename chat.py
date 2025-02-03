# This file handles the chat functionality.

from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

MODEL = "gpt-4o-mini"

def create_conversation_chain(vectorstore: Chroma):
    llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # the retriever is an abstraction over the VectorStore that will be used during RAG; k is how many chunks to use
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return conversation_chain

def chat(conversation_chain, message):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]
