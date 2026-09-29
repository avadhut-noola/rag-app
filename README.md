# rag-app
Building a RAG application by storing embeddings in Atlas, developing the retriever to perform a vector search, and creating a structured + clear prompt for the answer generator.

Preparing a PDF document for use with a Retrieval Augmented Generation (RAG) system. 

1. The PDF is split into chunks and stored in an Atlas Cluster. 
2. The chunks are then indexed using Atlas Vector Search. 
3. The indexed chunks can be used to retrieve relevant information for a given query.

# Prerequisites
1. Atlas Cluster Connection String
2. OpenAI API Key

# Usage
Install the requirements:
``` pip3 install langchain langchain_community langchain_core langchain_openai langchain_mongodb pymongo pypdf ```
Create a key_param file with the following content:
MONGODB_URI=<atlas_connection_string>
LLM_API_KEY=<llm_api_key>


# Load the sample data into your Atlas Cluster:
``` python load_data.py ```