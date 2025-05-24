"""
# LLM Demo using LangChain and OpenAI

This script demonstrates how to use the LangChain library with OpenAI's GPT-3.5-turbo-instruct model.
LLMs (Large Language Models) are powerful tools for natural language processing tasks.
LLMs take a string or plain text as an input and return a string or plain test as an output.
"""

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

