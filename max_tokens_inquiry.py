from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# The model you want to check
model_name = 'gpt-4'

# Retrieve model details
models = client.models.list()
# print(models)

# Find and print the token limit for the specified model
for model in models.data:
    print(model)
#     if model['id'] == model_name:
#         print(f"Model: {model['id']}")
        # print(f"Maximum tokens: {model.get('max_tokens', 'N/A')}")
#         break
# else:
#     print(f"Model {model_name} not found.")