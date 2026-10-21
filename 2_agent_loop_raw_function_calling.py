from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from google import genai
from google.genai import types
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gemini-3.6-flash"

client = genai.Client()


# --- Tools (Plain Python functions with LangSmith tracing) ---
# Difference 1: Without LangChain's @tool decorator, tools are regular Python functions.
# We use LangSmith's @traceable(run_type="tool") to keep monitoring tool executions.


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {
        "iphone 15": 1099,
        "macbook pro": 1999,
        "airpods pro": 249,
        "magic mouse": 79,
    }
    return prices.get(product.strip().lower(), 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: Bronze, Silver, Gold."""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"Bronze": 5, "Silver": 12, "Gold": 23}
    discount = discount_percentages.get(discount_tier.title(), 0)
    return round(price * (1 - discount / 100), 2)


# Difference 2: Without @tool, we must MANUALLY define the schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_product_price",
                description="Look up the price of a product in the catalog.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "product": {
                            "type": "STRING",
                            "description": "The product name, e.g. 'iphone 15', 'macbook pro', 'airpods pro'",
                        },
                    },
                    "required": ["product"],
                },
            ),
            types.FunctionDeclaration(
                name="apply_discount",
                description="Apply a discount tier to a price and return the final price. Available tiers: Bronze, Silver, Gold.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "price": {
                            "type": "NUMBER",
                            "description": "The original price",
                        },
                        "discount_tier": {
                            "type": "STRING",
                            "description": "The discount tier: 'Bronze', 'Silver', or 'Gold'",
                        },
                    },
                    "required": ["price", "discount_tier"],
                },
            ),
        ]
    )
]


# --- Helper: traced Google GenAI call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="Google GenAI Chat", run_type="llm")
def gemini_chat_traced(contents, generation_config):
    return client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=generation_config,
    )


# --- Agent Loop ---


@traceable(name="Google GenAI Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    # Difference 4: Native Content & Config objects instead of SystemMessage / HumanMessage
    system_instruction = (
        "You are a helpful shopping assistant. "
        "You have access to a product catalog tool "
        "and a discount tool.\n\n"
        "STRICT RULES - you must follow these exactly:\n"
        "1. Never guess or assume any product price. "
        "You must call get_product_price first to get the real price.\n"
        "2. Only call apply_discount AFTER you have received "
        "a price from get_product_price - do not pass a made-up number.\n"
        "3. Never calculate discounts yourself using math. "
        "Always use the apply_discount tool.\n"
        "4. If the user does not specify a discount tier, "
        "ask them which tier to use - do not assume one."
    )

    generation_config = types.GenerateContentConfig(
        temperature=0,
        tools=tools_for_llm,
        system_instruction=system_instruction,
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)],
        )
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # Difference 5: client.models.generate_content() directly instead of llm_with_tools.invoke()
        response = gemini_chat_traced(
            contents=contents, generation_config=generation_config
        )

        function_calls = response.function_calls

        # If no function calls, this is the final answer
        if not function_calls:
            print(f"\nFinal Answer: {response.text}")
            return response.text

        # Process only the FIRST tool call - force one tool per iteration
        tool_call = function_calls[0]
        # Difference 6: Attribute access (.name, .args) instead of dict access (.get("name"))
        tool_name = tool_call.name
        tool_args = tool_call.args

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)

        print(f"  [Tool Result] {observation}")

        # Append the model's response (containing the function call) to conversation history
        contents.append(response.candidates[0].content)

        # Append the tool execution result back to the model as a FunctionResponse
        tool_response_part = types.Part.from_function_response(
            name=tool_name,
            response={"result": observation},
        )
        contents.append(types.Content(role="user", parts=[tool_response_part]))

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Google GenAI Raw Agent Loop!")
    print()
    result = run_agent(
        "What is the price of a macbook pro and apply a gold discount on it?"
    )
