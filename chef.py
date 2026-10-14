from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
model = init_chat_model("google_genai:gemini-3.5-flash-lite")
# response = model.invoke("what can i cook with beaf and bell pepper")
# print(response.text)


# agent = create_agent(model=model, 
#                      system_prompt="you are a personal chef, Suggest recipes from the user's ingredients.")

#

tavily_client = TavilyClient()
@tool
def web_search(query: str) -> Dict[str, any]:
    """Search the web for recipes and cooking information"""
    return tavily_client.search(query)

agent = create_agent(model=model, tools = [web_search], system_prompt="you are a personal chef, Suggest recipes from the user's ingredients.")
response = agent.invoke({"messages": [HumanMessage(content="I have chicken and rice")]})
print(response["messages"][-1].text)
