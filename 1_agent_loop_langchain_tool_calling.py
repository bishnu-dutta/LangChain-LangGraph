from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
MODEL = "gemini-3.6-flash"

# ---- Tools (LangChain @tool dectorator) ---

@tool
def get_product_price(product: str) -> str:
    """look up the price of a product in the catalog."""
    print(f"Checking price for: {product}")
    prices = {
        "iphone 15": 1099,
        "macbook pro": 1999,
        "airpods pro": 249,
        "magic mouse": 79,
    }

    return prices.get(product.strip().lower(), 0)

@tool 
def apply_discount(price:float, discount_tier:str) -> float:
    """ Apply for a discount tier to a price and return the final price.
    Available tiers: Bronze, Silver, Gold. """
    print(f"Applying discount tier {discount_tier} to price {price}")
    discount_percentages = {
        "Bronze":5,
        "Silver":12,
        "Gold":23
    }
    discount = discount_percentages.get(discount_tier,0)
    final_price = price * (1- discount / 100)
    return round(final_price,2)
    

# --- Agent Loop ---

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}  

    llm = init_chat_model(model=MODEL, model_provider="google_genai")
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("="*60)

    messages = [
        # defensize prompting
        SystemMessage(
            content=(
                "You are a helpful shopping assistant"
                "You have access to a product catalog tool"
                "and a discount tool. \n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. Never guess or assume any product price."
                "You must call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received"
                "a price from get_product_price - do not pass a made-up number\n"
                "3. Never calculate discounts yourself using math"
                "Always use the apply_discount tool. \n"
                "4. If the user does not specify a discount tier,"
                "ask them which tier to use - do NOT assume ONE"
            ),
        ),
        HumanMessage(content=question),
    ]
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        messages.append(ai_message)

        # if not tool call, this is the final answer
        if not tool_calls:
            print("Final Answer:", ai_message.content)
            return ai_message.content

        # Process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' does not exist")
        
        observation = tool_to_use.invoke(tool_args)
        print(f"[Tool Output]: {observation}")

        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)        
        )  

    print("Error: Max Iterations reached without a final answer")
    return None          
    

if __name__ == "__main__":
    print("hello langchain agent (.bind_tools)! \n")
    result = run_agent("What is the price of a macbook pro and apply a gold discount on it?")
    