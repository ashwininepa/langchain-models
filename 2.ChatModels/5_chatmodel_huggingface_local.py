# import necessary libraries
import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


# ONLY FOR WINDOWS
os.environ['HF_HOME'] = "/Users/your_username/.cache/huggingface"  # Replace with your actual path

# Import the HuggingFacePipeline class from langchain_huggingface
llm = HuggingFacePipeline.from_model_id(
    model_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    pipline_kwargs=dict(
        temperature=0.6,
        max_new_tokens=50,
    )
)
# Initialize the ChatHuggingFace model with the HuggingFacePipeline
model = ChatHuggingFace(llm=llm)

# Invoke the model with a sample question
result = model.invoke('What is the capital of India?')

# Print the content of the result
print(result.content)