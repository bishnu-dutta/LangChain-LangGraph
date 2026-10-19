from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing import List

# ----------------- SCHEMA ------------------
class Source(BaseModel):
    "Schema for a source used for model"
    url : str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    "Schema for the agent response with answer and sources"
    answer:str = Field(description="The answer to the query")
    # if none default set to empty list
    sources : List[Source] = Field(default_factory=list, description="The list of sources used to generate the answer")

# ----------------- MODEL & TOOLS ------------------
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
tools = [TavilySearch()]

# using the AgentResponse Schema
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Its working")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain on linkedin and list their details")]})
    print(result)

if __name__ == "__main__":
    main()