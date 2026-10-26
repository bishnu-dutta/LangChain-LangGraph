import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("initializing componenets")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2", output_dimensionality=2048
)
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.environ["INDEX_NAME"],
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.
    
Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs):
    "Format retrieved docs into a single string"
    return "\n\n".join(document.page_content for document in docs)


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retieves documents, formats them and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No Built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # step 2: Format documnets into context string
    context = format_docs(docs)

    # step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # step 5: Return the Content
    return response.content


if __name__ == "__main__":
    print("retriving...")

    query = "what is hindu temple architecture"

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    response = llm.invoke([HumanMessage(content=query)])
    print(f"Answer: {response.content}")

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Retrieval chain without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print(f"Answer: {result_without_lcel}")
