from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime
model = init_chat_model("google_genai:gemini-3.5-flash-lite")

# 1. Describe what you know about the user
@dataclass 
class ChefContext:
    name: str = "odai"
    diet: str = "No tengo restricciones de dieta solo no aceite de todo excepto la aceite de oliva "
    allergies: list[str] = field(default_factory=list)
    servings: int = 2



# 2. Tools that read the context 
@tool
def get_dietary_needs(runtime: ToolRuntime[ChefContext]) -> str:
    """Obtener restricciones dietecticas y alergias del usario , siemper revisa eso antes sugerir recetas!"""
    ctx = runtime.context
    allergies = ", ".join(ctx.allergies) or "none"
    return f"Name: {ctx.name}. Diet {ctx.diet}. Allergies: {allergies}"

@tool 
def get_servings(runtime: ToolRuntime[ChefContext]) -> int:
    """Obtener para cuantos personsas vas a cocinar """
    return runtime.context.servings

# 3. Tell the agent which context to expect 
system_prompt = "Estas una chef personal, sugerir 3 recetas nombres que ajustado el usario necesidades y decir cuatnto comprar del ingrediente principal"
agent = create_agent(model = model, tools = [get_dietary_needs, get_servings], system_prompt=system_prompt, context_schema=ChefContext)

# 4. el mismo pregunta , dos usarios differentes
question = {"messages": [HumanMessage("yo tengo huevos, Cacahuates y fideos, que puedo prepara esta noche")]}
print("=== User 1: default settings ===")
r = agent.invoke(question , context= ChefContext())
print(r["messages"][-1].text)

print("\n=== User 2: vegano, allergia de Cacahuates, concinar por 6 ===")
r = agent.invoke(question, context=ChefContext(name="Sara", diet="vegan", allergies=["peanuts"], servings=6))
print(r["messages"][-1].text)





