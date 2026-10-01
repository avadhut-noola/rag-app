from pymongo import MongoClient
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

import key_param

# Set the MongoDB URI, DB, Collection Names

client = MongoClient(key_param.MONGODB_URI)
dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
collection = client[dbName][collectionName]

# Load PDF without langchain_community (avoids deprecation warning)
reader = PdfReader("./sample_files/mongodb.pdf")
pages = [
    Document(page_content=page.extract_text() or "", metadata={"page": i})
    for i, page in enumerate(reader.pages)
]
cleaned_pages = []

# Loop through the pages and clean the data
for page in pages:
    # If the page content is greater than 20 words, add it to the cleaned pages
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)

# Split the cleaned pages into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)


# Schema for metadata tagging (replaces deprecated openai_functions tagger)
class PageMetadata(BaseModel):
    title: str = Field(description="Short title for the page content")
    keywords: list[str] = Field(description="Relevant keywords")
    hasCode: bool = Field(description="Whether the page contains code samples")


# Calling the OpenAI API to tag the documents
llm = ChatOpenAI(
    openai_api_key=key_param.LLM_API_KEY, temperature=0, model="gpt-3.5-turbo"
)
structured_llm = llm.with_structured_output(PageMetadata)

# Tag each cleaned page with OpenAI-extracted metadata
docs = []
for page in cleaned_pages:
    meta = structured_llm.invoke(
        "Extract metadata for this document page:\n\n" + page.page_content
    )
    docs.append(
        Document(
            page_content=page.page_content,
            metadata={**page.metadata, **meta.model_dump()},
        )
    )

# Splitting the documents into chunks using the text splitter
split_docs = text_splitter.split_documents(docs)

# Creating the embeddings using the VoyageAI API
embeddings = VoyageAIEmbeddings(voyage_api_key=key_param.VOYAGE_API_KEY, model="voyage-3.5-lite")

# Creating the vector store using the MongoDB Atlas Vector Search
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    split_docs, embeddings, collection=collection
)
