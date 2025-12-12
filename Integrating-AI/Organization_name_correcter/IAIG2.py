from google import genai
from dotenv import load_dotenv
import os
import time
import tenacity

# Load environment variables
load_dotenv()

# Initialize the client
client = genai.Client()

# Retry logic for API calls
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),  # Retry 3 times
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff
    retry=tenacity.retry_if_exception_type(Exception)  # Retry on any exception
)
def call_generate_content():
    return client.models.generate_content(
        model="gemini-2.5-flash",
        contents="GTC Law Group PC & Affiliates and GTC Law Group PC - which organization name is correct? tell one",
    )

try:
    response = call_generate_content()
    print(response.text)
except Exception as e:
    print(f"Failed after retries: {e}")