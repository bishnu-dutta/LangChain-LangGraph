from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def get_weather(query:str) -> str:
    """
    Tool that simply take query and return query itself
    Args: The query to search for
    Returns: The search result
    """
    print(f"searching for {query}")
    return tavily.search(query=query)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
tools = [get_weather]

agent = create_agent(model=llm, tools=tools)


def main():
    print("Its working")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain on linkedin and list their details")]})
    print(result)

if __name__ == "__main__":
    main()