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
        "iPhone 15": 1099,
        "MacBook Pro": 1999,
        "AirPods Pro": 249,
        "Magic Mouse": 79,
    }

    return prices.get(product, 0)

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
    pass
    

if __name__ == "__main__":
    print("hello langchain agent (.bind_tools)! \n")
    result = run_agent("What is the price of a macbook pro and apply a gold discount on it")
    