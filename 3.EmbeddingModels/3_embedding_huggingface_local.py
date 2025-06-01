from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize the HuggingFace Embeddings model with the specified model and dimensions
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Sample text to be embedded
docs = [
    "Stockholm is the capital of Sweden.",
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy."
]

# Invoke the model with a sample query
embedding_vector = embedding.embed_documents(docs)

# Print the result of the embedding
print(str(embedding_vector))