from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)

import key_param

# Set the MongoDB URI, DB, Collection Names

client = MongoClient(key_param.MONGODB_URI)
dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
collection = client[dbName][collectionName]

loader = PyPDFLoader(".\sample_files\mongodb.pdf")
pages = loader.load()
cleaned_pages = []

# Loop through the pages and clean the data
for page in pages:
    # If the page content is greater than 20 words, add it to the cleaned pages
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)

# Split the cleaned pages into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)


# This is the schema for the metadata that will be used to tag the documents
schema = {
    "properties": {
        "title": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hasCode": {"type": "boolean"},
    },
    "required": ["title", "keywords", "hasCode"],
}

# Calling the OpenAI API to tag the documents
llm = ChatOpenAI(
    openai_api_key=key_param.LLM_API_KEY, temperature=0, model="gpt-3.5-turbo"
)

# Creating the document transformer
document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)

# Transforming the documents passed cleaned pages as argument
docs = document_transformer.transform_documents(cleaned_pages)

# Splitting the documents into chunks using the text splitter
split_docs = text_splitter.split_documents(docs)

# Creating the embeddings using the VoyageAI API
embeddings = VoyageAIEmbeddings(voyage_api_key=key_param.VOYAGE_API_KEY, model="voyage-3.5-lite")

# Creating the vector store using the MongoDB Atlas Vector Search
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    split_docs, embeddings, collection=collection
)