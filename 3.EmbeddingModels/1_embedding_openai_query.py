from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI Embeddings model with the specified model and dimensions
embedding = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions=32)

# Invoke the model with a sample query
result = embedding.embed_query("Stockholm is the capital of Sweden.")

# Print the result of the embedding
print(str(result))