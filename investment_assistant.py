import os
from anthropic import Anthropic

# Set your API key
api_key = os.getenv("CLAUDE_API_KEY")
if not api_key:
    print("Error: CLAUDE_API_KEY not set")
    exit(1)

client = Anthropic(api_key=api_key)

# Simple chat function
def chat(message):
    response = client.messages.create(
        model="claude-3-opus-4-7",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": message}
        ]
    )
    return response.content[0].text

# Test it
if __name__ == "__main__":
    result = chat("What is the capital of France?")
    print(result)
