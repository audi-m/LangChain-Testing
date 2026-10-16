import asyncio
import sys
from pathlib import Path 
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient({"chef_server": {
        "transport" : "stdio",
        "command" : sys.executable,
        "args" : ["chef_mcp_server.py"]
     }})

    # 1. Tools
    tools = await client.get_tools()
    print("Tools:", [t.name for t in tools])

    # 2. Resources
    resources = await client.get_resources("chef_server")
    print("\nResource:", resources[0].metadata["uri"])
    print("Content: ", resources[0].as_string())

    # 3. Prompts
    prompt = await client.get_prompt("chef_server", "chef_prompt")
    print("\nPrompt:\n", prompt[0].content)
    
    convert = tools[0]
    result = await convert.ainvoke({"amount": 2, "from_unit": "cups", "to_unit": "ml"})
    print("Result:", result)
asyncio.run(main())
