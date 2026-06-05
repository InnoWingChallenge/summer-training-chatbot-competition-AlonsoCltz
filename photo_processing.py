from dotenv import load_dotenv
import os
import chromadb
import os
from openai import AzureOpenAI
import base64
from pathlib import Path
import chromadb

BASE_DIR = Path.cwd().parent

load_dotenv(BASE_DIR / ".env")

API_Key = os.getenv("AZURE_OPENAI_API_KEY")

if not API_Key:
    raise RuntimeError("Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in .env or environment.")

client = AzureOpenAI(
    azure_endpoint="https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
    api_key=API_Key,
    api_version="2025-01-01-preview",
)

# Path to your image
image_path = "makerspace_A.jpeg"
# Read image and encode as base64
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe this image in detail."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    }
]

description = client.chat.completions.create(
    model="gpt-5-mini",
    messages=messages
).choices[0].message.content