from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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
            # k is the number of documents to return
            "k": 3,
            # pre_filter is used to filter the documents based on the given condition
            "pre_filter": { "hasCode": { "$eq": False } },
            
            # score_threshold is used to filter the documents based on the given score of relevance 0-1
            # 0.01 is the minimum score of relevance to return
        
            "score_threshold": 0.01
        },
    )

    # Prompt template for the RAG chain
    # 1. The context to use
    # 2. The question to answer
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Do not answer the question if there is no given context.
    Do not answer the question if it is not related to the context.
    Do not give recommendations to anything other than MongoDB.
    Context:
    {context}
    Question: {question}
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    # Retrieve the context from the vector store
    retrieve = {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
        "question": RunnablePassthrough()
        }

    # Initialize the LLM model
    # 1. The API key to use
    # 2. The temperature to use
    llm = ChatOpenAI(openai_api_key=key_param.LLM_API_KEY, temperature=0)

    # Parser for the response
    # 1. The response to parse
    response_parser = StrOutputParser()

    # RAG chain
    # 1. The context to use
    # 2. The question to answer
    # 3. The LLM model to use
    # 4. The response parser to use
    rag_chain = (
        retrieve
        | custom_rag_prompt
        | llm
        | response_parser
    )

    # Invoke the RAG chain by passing the query for the answer
    answer = rag_chain.invoke(query)
    

    return answer

# Test the RAG chain by passing a query for the answer
print(query_data("When did MongoDB begin supporting multi-document transactions?"))