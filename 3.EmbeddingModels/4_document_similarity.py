import enum
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI Embeddings model with the specified model and dimensions
embedding = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions=300)

# Sample documents to be embedded
docs = [
    "Stockholm is the capital of Sweden. It is known for its beautiful archipelago.",
    "Paris is the capital of France. It is famous for the Eiffel Tower.",
    "Berlin is the capital of Germany. It is known for its rich history and culture.",
    "Madrid is the capital of Spain. It is known for its vibrant nightlife.",
    "Rome is the capital of Italy. It is known for its ancient architecture and art."
]

query = "Tell me about the Germany."

# Embed the documents and the query
"""
This returns a 2D array of embeddings for the documents and a 1D array for the query.
Getting embeddings is a costly operation, so its important to store the results if you plan to use them multiple times.
Such results are stored in a vector database, such as Pinecone or Weaviate.

Here we are using the OpenAIEmbeddings model to embed the documents and the query and we are not storing the results in a vector database.
"""
docs_embeddings = embedding.embed_documents(docs) 
query_embedding = embedding.embed_query(query)

scores = cosine_similarity( # accepts a 2D array, so we need to reshape the query_embedding
    [query_embedding], 
    docs_embeddings
)[0]  # Get 1D array of scores

#print("Cosine Similarities:", scores)

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(docs[index])
print("Similarity Score: ", score)