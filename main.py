# This file is the main entry point and uses the other modules.

import gradio as gr
import numpy as np
from load_documents import load_documents
from vector_store import create_vector_store
from visualize_vector_store import visualize_2d, visualize_3d
from chat import create_conversation_chain, chat
from langchain.text_splitter import CharacterTextSplitter

# Load documents
documents = load_documents()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Total number of chunks: {len(chunks)}")
print(f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}")

# Create vector store
vectorstore = create_vector_store(chunks)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Visualize vector store
collection = vectorstore._collection
result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
doc_types = [metadata['doc_type'] for metadata in result['metadatas']]

# visualize_2d(vectors, doc_types, documents)
# visualize_3d(vectors, doc_types, documents)

# Create conversation chain
conversation_chain = create_conversation_chain(vectorstore)

# Test chat
query = "Can you describe Insurellm in a few sentences"
result = chat(conversation_chain, query)
print(result)

# Gradio interface
view = gr.ChatInterface(lambda message, history: chat(conversation_chain, message), type="messages").launch(inbrowser=True)