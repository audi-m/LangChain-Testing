from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langgraph.checkpoint.memory import InMemorySaver
import base64
from pathlib import Path
model = init_chat_model("google_genai:gemini-3.5-flash-lite")
# response = model.invoke("what can i cook with beaf and bell pepper")
# print(response.text)


# agent = create_agent(model=model, 
#                      system_prompt="you are a personal chef, Suggest recipes from the user's ingredients.")


tavily_client = TavilyClient()
@tool
def web_search(query: str) -> Dict[str, any]:
    """Search the web for recipes and cooking information"""
    return tavily_client.search(query)

#agent = create_agent(model=model, tools = [web_search], system_prompt="you are a personal chef, Suggest recipes from the user's ingredients.")
system_prom = "you are a personal chef, Suggest recipes from the user's ingredients."
agent = create_agent(model=model, tools = [web_search], system_prompt=system_prom )
config = {"configurable": {"thread_id": "1"}}
# agent.invoke({"messages": [HumanMessage("I have chicken and rice")]}, config)
 
img_b64 = base64.b64encode(Path("fridge.jpg").read_bytes()).decode("utf-8")
question = HumanMessage(content=[
    {"type": "text",  "text": "This is what's left in my fridge. What can I make?"},
    {"type": "image", "base64": img_b64, "mime_type": "image/jpeg"},
])
agent.invoke({"messages":[HumanMessage("Find me the best 10 recipe for these ingredients, Print Out in this format {Name - Country of origin}, no description nothing more list them in numbers")]}, config)
response  = agent.invoke({"messages":[HumanMessage("write recipe of number 1")]}, config)
response = agent.invoke({"messages": [question]}, config)
print(response["messages"][-1].text)