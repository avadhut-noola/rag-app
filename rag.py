from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings
import key_param

dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
index = "vector_index"

# Initialize the vector store by passing
# 1. The connection string to the MongoDB Atlas cluster
# 2. The name of the database and collection to use
# 3. The embeddings model to use
# 4. The name of the index to use which is created in the MongoDB Atlas cluster
vectorStore = MongoDBAtlasVectorSearch.from_connection_string(
    key_param.MONGODB_URI,
    dbName + "." + collectionName,
    VoyageAIEmbeddings(voyage_api_key=key_param.VOYAGE_API_KEY, model="voyage-3.5-lite"),
    index_name=index,
)

# Define a function to query the data from the vector store
# 1. The query to search for
# 2. The type of search to perform
# 3. The number of results to return
def query_data(query):
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3
        },
    )

    results = retriever.invoke(query)
    print(results) 