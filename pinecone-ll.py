import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Replace with whatever index name you want to use
INDEX_NAME = "medium-blogs-standard-index-test"
pc.create_index(
    name=INDEX_NAME,
    dimension=2048,  # matches your Google Gemini output_dimensionality=2048
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",  # Free tier default region
    ),
)
print(f"Created standard vector index: {INDEX_NAME}")
