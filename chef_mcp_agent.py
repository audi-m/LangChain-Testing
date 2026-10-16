import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
model = init_chat_model("google_genai:gemini-3.5-flash-lite")


async def main():
    # 1. connect to your MCP server 
    server_file = Path(__file__).parent / "chef_mcp_server.py"
    client = MultiServerMCPClient({
        "chef_server": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(server_file)]
        }
    })
    # 2. Get everything the server offers 
    tools = await client.get_tools()
    resources = await client.get_resources("chef_server")
    prompt = await client.get_prompt("chef_server", "chef_prompt")
    pantry = resources[0].as_string()
    system_prompt = prompt[0].content + f"\n The user always have these at home {pantry}"

    # 3. Build the agent from the servers tools and prompt 
    agent = create_agent(model = model, tools = tools, system_prompt=system_prompt, checkpointer=InMemorySaver())
    config = {"configurable" : {"thread_id": "1"}}
    seen = 0 

    #4. chat loop
    print("chef ready type quit to exist \n")
    while True: 
        user = input("you: ").strip()
        if user.lower() in ("quit" , "exit"):
            break
        response = await agent.ainvoke({"messages": HumanMessage(user)}, config)
         # show which tools the agent decided to use THIS turn
        for msg in response["messages"][seen:]:
           for call in getattr(msg, "tool_calls", None) or []:
                print(f"   [tool] {call['name']}({call['args']})")
        seen = len(response["messages"])

        print(f"\nChef: {response['messages'][-1].text}\n")

asyncio.run(main())
