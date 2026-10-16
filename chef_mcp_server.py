from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
mcp = FastMCP("chef_server")
tavily_client = TavilyClient()
@mcp.tool()
def convert_units(amount: float, from_unit: str, to_unit: str) -> str :
    """Convert cooking quantities between units: cups, tbsp, tsp, ml, g, oz, lb"""
    to_ml = {"cup": 240, "tbsp": 15, "tsp": 5, "ml": 1}
    to_g = {"g": 1, "oz": 28.35, "lb": 453.6}
    f = from_unit.lower().rstrip("s")
    t = to_unit.lower().rstrip("s")
    for table in (to_ml, to_g):
        if f in table and t in table:
            result = round(amount * table[f] / table[t], 2)
            return f"{amount} {from_unit} = {result} {to_unit}"
    return f"Can't convert {from_unit} to {to_unit}"

@mcp.tool()
def search_recipes(query: str) -> Dict[str, Any]:
    """Search the web for recipes"""
    return tavily_client.search(query)


@mcp.resource("kitchen://pantry-basics")
def pantry_basics() -> str:
    """Staples the user always has at home"""
    return "salt, black pepper, olive oil, butter, flour, sugar, garlic, onions, rice, pasta"


@mcp.prompt()
def chef_prompt() -> str:
    """System prompt for a personal chef agent"""
    return """You are a personal chef.

You can use these tools:
- search_recipes: find real recipes on the web
- convert_units: convert cooking measurements

First suggest 3-5 recipe NAMES with cooking time. When the user picks one,
give the full recipe with ingredients, numbered steps and the source link.

If the user asks about anything that isn't food or cooking, say:
"I'm sorry, I can only help with cooking."
"""
if __name__ == "__main__":
    mcp.run(transport="stdio")