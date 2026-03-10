import json
import os
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from db_connection import get_mysql_connection as db_engine

load_dotenv()

# 1. Initialize database (using SQLDatabase utility for convenience in schema/execution)
db = SQLDatabase(db_engine())
llm = ChatOllama(model="qwen2.5", temperature=0)
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- CUSTOM TOOLS ---


@tool
def sql_execution_tool(query_intent: str) -> str:
    """
    Given a user's natural language intent, fetches the schema, 
    generates a safe SQL query, and executes it against the database.
    """
    # 1. Fetch Schema
    schema = db.get_table_info()
    
    # 2. Prompt model to generate SQL based on schema
    prompt = f"""
    You are a MySQL expert. Based on the following schema, generate a valid MySQL SELECT query that fulfills the user's intent.
    Output ONLY THE RAW SQL, no markdown, no explanation.
    
    SCHEMA:
    {schema}
    
    USER INTENT:
    {query_intent}
    """
    
    # We use the raw LLM here to get just the string
    generated_sql = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    
    # Remove markdown formatting if present
    if "```sql" in generated_sql:
        generated_sql = generated_sql.split("```sql")[1].split("```")[0].strip()
    elif "```" in generated_sql:
        generated_sql = generated_sql.split("```")[1].split("```")[0].strip()

    # 3. Validate for forbidden keywords
    forbidden = ["delete", "drop", "update", "insert", "truncate", "alter"]
    for word in forbidden:
        if word in generated_sql.lower():
            return f"Error: Forbidden keyword '{word}' detected in generated query. Execution denied."

    # 4. Execute
    try:
        # print(f"Executing SQL: {generated_sql}")
        result = db.run(generated_sql)
        # Return as JSON string so it's easy for subsequent tools to parse
        return result
    except Exception as e:
        return f"Error executing SQL: {e}"

@tool
def generate_report_tool(data: str, filename: str = "report.xlsx") -> str:
    """
    Generates an Excel report from database results.
    'data' can be a list of dictionaries or a JSON-formatted string.
    """
    try:
        import ast
        import json
        
        # If it's already a list or dict, use it directly
        if isinstance(data, (list, dict)):
            df = pd.DataFrame(data)
        elif isinstance(data, str):
            try:
                # Try JSON first
                parsed_data = json.loads(data)
                df = pd.DataFrame(parsed_data)
            except json.JSONDecodeError:
                # Try literal_eval if it's a string representation of a list of tuples
                try:
                    parsed_data = ast.literal_eval(data)
                    df = pd.DataFrame(parsed_data)
                except Exception:
                    return f"Error: Could not parse string data. Data: {data[:100]}..."
        else:
            return f"Error: Unsupported data type {type(data)}."

        if df.empty:
            return "No data found to generate report."
            
        df.to_excel(filename, index=False)
        return f"Report successfully generated and saved to {filename}."
    except Exception as e:
        return f"Error generating report: {e}"

# --- LANGGRAPH SETUP ---
tools = [sql_execution_tool, generate_report_tool]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """
You are a helpful assistant for a school database.
You have access to:
1. `sql_execution_tool`: Use this to fetch data for ANY user question about students, marks, etc.
2. `generate_report_tool`: Use this ONLY if the user explicitly asks for an Excel file or a report. 
   Input the EXACT string returned by `sql_execution_tool` into `generate_report_tool`.

ALWAYS start with `sql_execution_tool` if the user asks a question about data.
NOTE: Note that whatever data you show that should be in table format with proper columns and rows.
"""

def call_model(state: MessagesState):
    response = llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )
    return {"messages": [response]}

def human_approval_node(state: MessagesState):
    """
    Node that checks if we just finished sql_execution_tool and need approval for report.
    """
    last_message = state["messages"][-1]
    human_message = state["messages"][0]

    # 2. Prompt to generate report generation question
    prompt = f"""
    You are a expert in identifying the user intent. Based on the following Human Message, generate a question wheather user want excel report or not.

    Human Message:
    {human_message.content}    
    """

    hitl_prompt = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    
    # Check if the last tool executed was sql_execution_tool
    if isinstance(last_message, ToolMessage) and last_message.name == "sql_execution_tool":
        user_input = input(f"\n[HITL] {hitl_prompt}: ").lower().strip()
        promt = f"""
            You are a expert in identifying the user intent.
            Based on the following User Concent, read the asked question by agent and generate a response either YES or NO

            Question Asked by Agent:
            {hitl_prompt}

            User Concent:
            {user_input}
        """

        human_approval_response = llm.invoke([HumanMessage(content=promt)]).content.strip()

        if "yes" in human_approval_response.lower():
            return {"messages": [HumanMessage(content="Yes, please generate an Excel report with this data.")]}
        else:
            return {"messages": [HumanMessage(content="No, I don't want a report. End the workflow.")]}
    
    return {}

def route_after_agent(state: MessagesState):
    """
    Custom router to handle HITL logic.
    """
    # Standard tools_condition logic first
    route = tools_condition(state)
    if route == "tools":
        # Check if the tool being called is sql_execution_tool
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls"):
            tool_calls = last_msg.tool_calls
            # If any tool call is sql_execution_tool, we go to tools normally
            # The HITL logic will trigger AFTER tools node finishes
            return "tools"
    return route

def route_after_tools(state: MessagesState):
    """
    Router after tools node execution.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage) and last_message.name == "sql_execution_tool":
        return "human_approval"
    return "agent"

def route_after_human(state: MessagesState):
    """
    Router after human input.
    """
    last_message = state["messages"][-1]
    if "Yes, please generate" in last_message.content:
        return "agent" # Let agent decide to call generate_report_tool
    return "__end__"

# Build the graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("human_approval", human_approval_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "__end__": END})
workflow.add_conditional_edges("tools", route_after_tools, {"human_approval": "human_approval", "agent": "agent"})
workflow.add_conditional_edges("human_approval", route_after_human, {"agent": "agent", "__end__": END})

app = workflow.compile()
query="Show me list of 10 students?"


# --- EXECUTION ---
if __name__ == "__main__":
    response = app.invoke({"messages": [HumanMessage(content=query)]})
    print(response)
    
# Show workflow
try:
    png_data = app.get_graph().draw_mermaid_png()
    with open("./database_workflow.png", "wb") as f:
        f.write(png_data)
except Exception:
    pass # Ignore error in terminal environments
