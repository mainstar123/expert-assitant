# This file handles creating and managing the vector store.

import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

db_name = "vector_db"

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()

    # Delete if already exists
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

    # Create vectorstore
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
    return vectorstore
