# Import necessary libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize the ChatGoogleGenerativeAI model with the specified model version
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Invoke the model with a sample question
result = model.invoke('What is the capital of India?')

# Print the content of the result
print(result.content)