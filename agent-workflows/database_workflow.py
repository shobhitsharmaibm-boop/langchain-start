# from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from db_connection import get_mysql_connection as db_engine

load_dotenv()

SYSTEM_PROMPT = """
    You are a MySQL expert for a school database. 
    - Tables: attendance, marks, students, subjects, teachers.
    - Always use MySQL syntax.
    - Verify schema and table structure with their relationship with other tables.
    - Analyze the relational structure by inspecting foreign key constraints between tables.
    - Use SQL JOIN operations (INNER JOIN, LEFT JOIN) whenever a query requires data spanning multiple tables.
    - When retrieving student performance or attendance, ensure you join `students` with `marks` or `attendance` using the appropriate keys.
    - Always prefer human-readable names (e.g., student names, subject titles) over internal IDs by joining the relevant reference tables.
    - Verify the join keys match the schema's foreign key definitions to ensure query accuracy.
    - For performance analysis, use aggregate functions like AVG() or SUM() in conjunction with GROUP BY on student or subject names.
    - When querying teacher information, link `teachers` with `subjects` to identify which teacher handles which course.
"""

# 1. Initialize database and toolkit
db = SQLDatabase(db_engine())

# 2. Initialize model and bind tools
llm = ChatOllama(model="qwen2.5", temperature=0)
# llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
llm_with_tools = llm.bind_tools(tools)

# 3. Define the node that calls the model
def call_model(state: MessagesState):
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
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
response = app.invoke({"messages": [("user", "Show me attendance percentage of all students of Class 9th?")]})

print(response["messages"][-1].content)
# print("\n--- Message Trace ---")
# for msg in response["messages"]:
#     print(f"{msg.__class__.__name__}: {msg.content}")

