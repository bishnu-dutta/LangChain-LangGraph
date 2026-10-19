from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from tavily import TavilyClient


@tool
def get_weather(query:str) -> str:
    """
    Tool that simply take query and return query itself
    Args: The query to search for
    Returns: The search result
    """
    print(f"searching for {query}")
    return f"The weather there is sunny"

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
tools = [get_weather]

agent = create_agent(model=llm, tools=tools)


def main():
    print("Its working")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in tokyo?")]})
    print(result)

if __name__ == "__main__":
    main()