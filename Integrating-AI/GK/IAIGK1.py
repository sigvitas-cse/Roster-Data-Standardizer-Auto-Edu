import requests
import os
from dotenv import load_dotenv

# Load API key from .env file (create a .env file with XAI_API_KEY=your_actual_key)
load_dotenv()

url = "https://api.x.ai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}",
    "Content-Type": "application/json"
}
data = {
    "messages": [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Testing. Just say hi and hello world and nothing else."}
    ],
    "model": "grok-4-latest",
    "stream": False,
    "temperature": 0
}

response = requests.post(url, json=data, headers=headers)
if response.status_code == 200:
    print(response.json()["choices"][0]["message"]["content"])
else:
    print(f"Error: {response.status_code} - {response.text}")