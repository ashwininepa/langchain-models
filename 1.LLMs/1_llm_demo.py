## Import necessary libraries
from langchain_openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI LLM with a specific model
llm = OpenAI(model="gpt-3.5-turbo-instruct")

# Invoke the LLM with a sample question
result = llm.invoke("What is the capital of India?")

# Print the result
print(result)

