# from typing import Annotated, Literal, TypedDict
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display


# 1. Define the tools
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts b from a."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

tools = [add, subtract, multiply]

# 2. Initialize model and bind tools
llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 3. Define the node that calls the model
def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    # We return a list, because MessagesState appends new messages
    return {"messages": [response]}

# 4. Build the graph
workflow = StateGraph(MessagesState)

# Add the LLM node and the prebuilt ToolNode
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# Set the entrypoint
workflow.add_edge(START, "agent")

# Use the prebuilt tools_condition to route to "tools" if the model
# called a tool, or to END if it responded with text.
workflow.add_conditional_edges("agent", tools_condition)

# Always go back to the agent after executing tools to process results
workflow.add_edge("tools", "agent")

app = workflow.compile()

# Example invocation
response = app.invoke({"messages": [("user", "What is (3 + 5) * 2 - 4?")]})

print("\n--- Message Trace ---")
for msg in response["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content}")

# Replace 'app' with your compiled graph variable
try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    pass # Ignore error in terminal environments