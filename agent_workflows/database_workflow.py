import sys
import os

sys.path.append(os.getcwd())


import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
import uuid

from db_connection import get_mysql_connection as db_engine

load_dotenv()

# 1. Initialize database (using SQLDatabase utility for convenience in schema/execution)
db = SQLDatabase(db_engine())
# llm = ChatOllama(model="qwen3.5", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def parse_data_to_df(input_data) -> pd.DataFrame:
    """Converts input data (list, dict, or string) into a pandas DataFrame.
    Supports JSON and literal evaluation for string inputs."""
    if isinstance(input_data, (list, dict)):
        return pd.DataFrame(input_data)
    elif isinstance(input_data, str):
        try:
            import json

            return pd.DataFrame(json.loads(input_data))
        except json.JSONDecodeError:
            try:
                import ast

                return pd.DataFrame(ast.literal_eval(input_data))
            except Exception:
                raise ValueError(
                    f"Could not parse string data. Data: {input_data[:100]}..."
                )
    raise TypeError(f"Unsupported data type {type(input_data)}.")


# --- CUSTOM TOOLS ---
@tool
def convert_table_format_tool(data: str) -> str:
    """
    Converts data to table format.
    """
    try:
        from tabulate import tabulate

        df = parse_data_to_df(data)
        return tabulate(df, headers="keys", tablefmt="grid", showindex=False)

    except Exception as e:
        return f"Error converting SQL to DataFrame: {e}"


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
        df = parse_data_to_df(data)

        if df.empty:
            return "No data found to generate report."

        df.to_excel(filename, index=False)
        return f"Report successfully generated and saved to {filename}."
    except Exception as e:
        return f"Error generating report: {e}"


SYSTEM_PROMPT = """
You are a helpful assistant for a school database.
You have access to:
1. `sql_execution_tool`: Use this to fetch data for ANY user question about students, marks, etc.
2. `generate_report_tool`: Use this ONLY if the user explicitly asks for an Excel file or a report. 
   Input the EXACT string returned by `sql_execution_tool` into `generate_report_tool`.

ALWAYS start with `sql_execution_tool` if the user asks a question about data.

"""

# --- LANGGRAPH SETUP ---
tools = [sql_execution_tool, generate_report_tool, convert_table_format_tool]
middleware = [
    HumanInTheLoopMiddleware(
        interrupt_on={
            "generate_report_tool": True,
            "sql_execution_tool": False,
            "convert_table_format_tool": False,
        }
    )
]
agent = create_agent(
    model=llm, tools=tools, system_prompt=SYSTEM_PROMPT, middleware=middleware
)


def agent_node(state: MessagesState) -> MessagesState:
    """LangGraph node calling your HITL agent."""
    result = agent.invoke(state)
    return {"messages": [AIMessage(content=result["output"])]}


# Build the graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}
app = workflow.compile(checkpointer=InMemorySaver())

# --- EXECUTION ---
if __name__ == "__main__":

    response = app.invoke(
        input={"messages": [HumanMessage(content="Show me list of 10 students")]},
        config=config,
    )
    print("Graph state:", response["messages"][-1].content)

    n = app.invoke(
        input={
            "messages": [
                HumanMessage(
                    content="Add student class name as well in the student list"
                )
            ]
        },
        config=config,
    )
    print("Graph state:", n["messages"][-1].content)

    response2 = app.invoke(
        input={"messages": [HumanMessage(content="Yes, generate excel report")]},
        config=config,
    )
    print("Graph state:", response2["messages"][-1].content)

    if "__interrupt__" in response2:
        interrupt = response2["__interrupt__"][0].value
        print("Pending actions:", interrupt["action_requests"])

        # Human decision (one per action, same order)
        response3 = app.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),  # or "reject", "edit"
            config=config,
        )
        print("Final result:", response3["messages"][-1].content)

# Show workflow
try:
    png_data = app.get_graph().draw_mermaid_png()
    with open("./database_workflow.png", "wb") as f:
        f.write(png_data)
except Exception:
    pass  # Ignore error in terminal environments
