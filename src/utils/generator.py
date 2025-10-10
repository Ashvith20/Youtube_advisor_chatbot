# generator.py

from dotenv import load_dotenv
import os
from groq import Groq

# Load environment variables from .env
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_TOKEN")

# Initialize Groq client
client = Groq(api_key=api_key)


def generate_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=200
    )
    return response.choices[0].message.content
