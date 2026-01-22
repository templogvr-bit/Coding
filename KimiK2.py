import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Initialize client
client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),  # Please ensure environment variable is set
    base_url="https://api.moonshot.ai/v1"
)

# Make chat completion request
try:
    response = client.chat.completions.create(
        model="kimi-k2-0905-preview",
        messages=[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  }
],
        temperature=0.3,
        max_tokens=8192,
        top_p=1,
        stream=True
    )
    
    # Handle streaming response
    for chunk in response:
        choice = chunk.choices[0]
        if choice.delta and hasattr(choice.delta, "reasoning_content"):
            reasoning_content = getattr(choice.delta, "reasoning_content", None)
            if reasoning_content:
                print(reasoning_content, end="")
        if choice.delta and choice.delta.content is not None:
            print(choice.delta.content, end="")
    print()  # New line
    
except Exception as e:
    print(f"Request failed: {e}")
