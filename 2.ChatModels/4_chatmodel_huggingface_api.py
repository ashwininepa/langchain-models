from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize the HuggingFace model with the specified endpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    huggingfacehub_api_token="your_huggingface_api_token_here",  # Replace with your Hugging Face API token
)
model = ChatHuggingFace(llm=llm)

# Invoke the model with a sample question
result = model.invoke('What is the capital of India?')

# Print the content of the result
print(result.content)