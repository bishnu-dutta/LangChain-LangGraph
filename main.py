import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("ingestion")
    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    texts = text_splitter.split_documents(document)
    print(f"Split {len(texts)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        output_dimensionality=2048,
    )

    print("ingestion...")
    PineconeVectorStore.from_documents(
        texts,
        embedding=embeddings,
        index_name=os.getenv("INDEX_NAME"),
    )
    print("finish")
