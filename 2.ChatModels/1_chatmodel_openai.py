"""
# LLM Demo using LangChain and ChatOpenAI
This script demonstrates how to use the LangChain library with OpenAI's GPT-4.1-nano model.
"""

# Load libraries
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI Chat Model with a specific model
"""
Parameters used here:
- 'model' specifies the OpenAI model to use, in this case, "gpt-4.1-nano-2025-04-14".
- 'temperature' parameter controls the randomness of the output, typically between 0.0 and 2.0 but can sometimes be higher.
Lower temperature (< 1.0) makes the output more deterministic,
while higher temperature (> 1.0) makes it more creative and varied.
- 'max_completion_tokens' limits the number of tokens in the response.
"""
model = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0, max_completion_tokens=10)

# Invoke the Chat Model with a sample question
result = model.invoke("Write a poem about the beauty of nature in springtime.")

# Print the result
print(result.content)