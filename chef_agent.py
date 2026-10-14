"""
Personal Chef Agent (LangGraph) — reads a photo of your fridge,
suggests recipe names, and gives full details for the one you pick.

Runs on Google Gemini's FREE tier (no credit card):
    pip install -U langgraph langchain-google-genai python-dotenv
    # get a key at https://aistudio.google.com/apikey
    export GOOGLE_API_KEY="..."
    python chef_agent.py

In the chat:
    photo fridge.jpg          -> scans the photo and suggests recipes
    2   (or a recipe name)    -> full recipe for that suggestion
    I'm vegetarian            -> chef remembers it
    quit

Optional: CHEF_MODEL=gemini-3.5-flash-lite (any free-tier Gemini model that accepts images)
"""

import os
import base64
import mimetypes
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL = os.getenv("CHEF_MODEL", "gemini-3.6-flash")


# ---------------------------------------------------------------------------
# STEP 1 — State: chat messages + the photo + what the chef saw in the fridge
# ---------------------------------------------------------------------------

class ChefState(MessagesState):
    photo_b64: Optional[str]          # the new photo (cleared after scanning)
    photo_mime: Optional[str]
    fridge_items: list[str]           # ingredients detected in the latest photo


# ---------------------------------------------------------------------------
# STEP 2 — Vision: turn a fridge photo into a clean ingredient list
# ---------------------------------------------------------------------------

class FridgeScan(BaseModel):
    """Ingredients visible in a fridge photo."""
    items: list[str] = Field(description="Each distinct food item, e.g. '3 eggs', 'half a red bell pepper', 'plain yogurt'")
    notes: str = Field(default="", description="Anything that looks spoiled or almost finished")


VISION_PROMPT = (
    "You are looking at a photo of someone's fridge or food shelf. "
    "List every food item you can identify, with a rough quantity when visible. "
    "Read labels on jars and cartons. Do not guess items you cannot see. "
    "Ignore non-food objects."
)


def make_vision_fn():
    from langchain_google_genai import ChatGoogleGenerativeAI
    vision = ChatGoogleGenerativeAI(model=MODEL, temperature=0).with_structured_output(FridgeScan)

    def scan(photo_b64: str, mime: str) -> FridgeScan:
        msg = HumanMessage(content=[
            {"type": "text", "text": VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{photo_b64}"}},
        ])
        return vision.invoke([msg])
    return scan


# ---------------------------------------------------------------------------
# STEP 3 — Tools the chef can call (memory of your preferences)
# ---------------------------------------------------------------------------

USER_PROFILE: dict[str, list[str]] = {"diet": [], "allergies": [], "dislikes": []}


@tool
def save_preference(category: str, value: str) -> str:
    """Remember a dietary preference. category must be one of: diet, allergies, dislikes."""
    category = category.lower().strip()
    if category not in USER_PROFILE:
        return "Unknown category. Use diet, allergies or dislikes."
    if value.lower() not in USER_PROFILE[category]:
        USER_PROFILE[category].append(value.lower())
    return f"Saved. Profile: {USER_PROFILE}"


TOOLS = [save_preference]


# ---------------------------------------------------------------------------
# STEP 4 — The chef's instructions: names first, details on request
# ---------------------------------------------------------------------------

def chef_prompt(fridge_items: list[str]) -> str:
    fridge = ", ".join(fridge_items) if fridge_items else "(no photo scanned yet)"
    return f"""You are Chef Remy, a warm, practical personal chef.

What's in the user's fridge (from their latest photo): {fridge}
User profile: {USER_PROFILE}

How you work:
1. Right after a new fridge scan, briefly list what you spotted, then suggest
   4-5 recipes as a NUMBERED list. For each give ONLY: the recipe name,
   cooking time, and one short line on why it fits. No ingredients or steps yet.
   End by asking which one they'd like.
2. When the user picks one (by number or name), give the FULL recipe:
   servings, total time, ingredients split into "From your fridge" and
   "You may need" (pantry basics like salt, oil, spices), numbered steps,
   and one chef's tip.
3. Prefer recipes that use what's in the fridge, especially items about to expire.
4. If the user mentions a diet, allergy or dislike, call save_preference.
   Never suggest anything containing a saved allergen.
5. If no photo has been scanned, ask the user to send one (type: photo <path>)."""


def make_chef_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=MODEL, temperature=0.5).bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# STEP 5 — The graph
#
#   START ─┬─(photo?)─> scan_fridge ─> chef ─┬─(tool call?)─> tools ─> chef
#          └──────────────────────────> chef ┴─> END
# ---------------------------------------------------------------------------

def build_graph(chef_llm=None, vision_fn=None):
    chef_llm = chef_llm or make_chef_llm()
    vision_fn = vision_fn or make_vision_fn()

    def route_start(state: ChefState) -> str:
        return "scan_fridge" if state.get("photo_b64") else "chef"

    def scan_fridge(state: ChefState):
        result = vision_fn(state["photo_b64"], state.get("photo_mime") or "image/jpeg")
        items = list(result.items)
        note = f" Notes: {result.notes}" if result.notes else ""
        return {
            "fridge_items": items,
            "photo_b64": None,  # don't resend the image every turn
            "messages": [HumanMessage(f"[New fridge scan] I found: {', '.join(items)}.{note} "
                                      "Suggest recipe names I can make.")],
        }

    def chef(state: ChefState):
        msgs = [SystemMessage(chef_prompt(state.get("fridge_items") or []))] + state["messages"]
        return {"messages": [chef_llm.invoke(msgs)]}

    builder = StateGraph(ChefState)
    builder.add_node("scan_fridge", scan_fridge)
    builder.add_node("chef", chef)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_conditional_edges(START, route_start, ["scan_fridge", "chef"])
    builder.add_edge("scan_fridge", "chef")
    # tools_condition returns "tools" or END on its own, so no mapping is needed
    builder.add_conditional_edges("chef", tools_condition)
    builder.add_edge("tools", "chef")

    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# STEP 6 — Helpers + terminal chat loop
# ---------------------------------------------------------------------------

def load_photo(path: str) -> tuple[str, str]:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def ask_chef(graph, config, text: str = "", photo_b64: str = None, mime: str = None) -> str:
    update = {"messages": [HumanMessage(text)] if text else []}
    if photo_b64:
        update.update(photo_b64=photo_b64, photo_mime=mime)
    result = graph.invoke(update, config)
    content = result["messages"][-1].content
    if isinstance(content, list):  # Gemini can return content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "my-kitchen"}}
    print("Chef Remy is ready! Type 'photo <path-to-image>' to scan your fridge. 'quit' to exit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in ("quit", "exit", "q"):
            break
        if not user:
            continue
        if user.lower().startswith("photo "):
            path = user[6:].strip().strip('"\'')
            if not os.path.exists(path):
                print(f"Can't find {path}\n")
                continue
            b64, mime = load_photo(path)
            print("Chef: Looking in your fridge...\n")
            reply = ask_chef(graph, config, photo_b64=b64, mime=mime)
        else:
            reply = ask_chef(graph, config, text=user)
        print(f"Chef: {reply}\n")


if __name__ == "__main__":
    main()