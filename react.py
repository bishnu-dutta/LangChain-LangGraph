from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def triple(num: float) -> float:
    """param num: a nuumber to triple
    returns: the triple of the input number
    """
    return float(num * 3)


tools = [TavilySearch(max_results=1), triple]

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0).bind_tools(tools)
