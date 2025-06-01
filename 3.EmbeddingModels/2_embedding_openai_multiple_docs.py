from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI Embeddings model with the specified model and dimensions
embedding = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions=32)

# Sample documents to be embedded
docs = [
    "Stockholm is the capital of Sweden.",
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy."
]

# Invoke the model with a sample query
result = embedding.embed_documents(docs)

# Print the result of the embedding
print(str(result))