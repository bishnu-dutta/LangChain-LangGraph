import os

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    print("ingestion")
    print(os.environ["PINECONE_API_KEY"])
