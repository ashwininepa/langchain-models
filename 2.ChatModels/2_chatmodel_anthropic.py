# Import necessary libraries
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize the ChatAnthropic model with the specified model version
model = ChatAnthropic(model="claude-3.5-sonnet-20241022")

# Invoke the model with a sample question
result = model.invoke('What is the capital of India?')

# Print the content of the result
print(result.content)