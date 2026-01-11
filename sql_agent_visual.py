
import os
import getpass
from typing import Literal, Annotated
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Set your API keys here or in your environment variables
# Note: GOOGLE_API_KEY is not needed when using OpenRouter
# (Tracing is now controlled by .env)
# --- Configuration ---
# API Keys are loaded from .env
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print("CRITICAL ERROR: OPENROUTER_API_KEY is missing from environment!")
    print("Please make sure you have a .env file with OPENROUTER_API_KEY=sk-or-...")
else:
    print(f"DEBUG: Found OPENROUTER_API_KEY starting with {api_key[:5]}...")
    api_key = api_key.strip()
    # ChatOpenAI often defaults to looking for OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = api_key

# --- LangSmith Compatibility ---
# The LangSmith UI sometimes recommends LANGSMITH_ prefixes, but the library expects LANGCHAIN_ prefixes.
# We map them here for convenience.
for key in ["TRACING", "ENDPOINT", "API_KEY", "PROJECT"]:
    ls_key = f"LANGSMITH_{key}"
    lc_key = f"LANGCHAIN_{key if key != 'TRACING' else 'TRACING_V2'}"
    if os.environ.get(ls_key) and not os.environ.get(lc_key):
        os.environ[lc_key] = os.environ[ls_key]
        print(f"DEBUG: Mapped {ls_key} to {lc_key}")

if not os.environ.get("LANGCHAIN_API_KEY"):
    pass

# --- Imports ---
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

# 1. Initialize Model (OpenRouter)
model = ChatOpenAI(
    model="deepseek/deepseek-chat",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    # Some versions prefer base_url
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

# 2. Connect to Database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(f"Database Dialect: {db.dialect}")
print(f"Usable Tables: {db.get_usable_table_names()}")

# 3. Setup SQL Tools
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")

# 4. Setup Visualization Tool (PythonREPL)
python_repl = PythonREPL()

def run_python_repl(code: str):
    return python_repl.run(code)

python_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. Useful for visualization with matplotlib/seaborn. When plotting, save text/result or just let it start.",
    func=run_python_repl
)

# --- Agent Functions & Nodes ---

def list_tables(state: MessagesState):
    """Refreshes the table list context."""
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "init_tables",
        "type": "tool_call",
    }
    # Create the interaction chain manually so it appears in history
    ai_msg = AIMessage(content="Checking database tables...", tool_calls=[tool_call])
    tool_msg = list_tables_tool.invoke(tool_call)
    
    return {"messages": [ai_msg, tool_msg]}

def generate_query(state: MessagesState):
    """Decides on the next action: Query SQL or Visualize Data."""
    
    prompt = """
    You are a sophisticated SQL + Data Visualization Agent.
    
    Your goal is to answer the user's question using the `Chinook` database.
    
    GUIDELINES:
    1. **Data Retrieval**: Always start by writing a valid SQLite query to get the necessary data. Use `sql_db_query`.
    2. **Visualization**: If the user asks for a chart/graph/plot:
       - FIRST, fetch the data with SQL.
       - SECOND, once you have the data (it will appear in your history), use `python_repl` to plot it.
       - In your Python code, manually define the data lists based on the SQL result.
       - Example Python Code:
         ```python
         import matplotlib.pyplot as plt
         data = [10, 20, 30]
         labels = ['A', 'B', 'C']
         plt.bar(labels, data)
         print("Chart created.")
         ```
    3. **Schema**: If you are unsure about column names, use `sql_db_schema`.
    
    Current Tables: {tables}
    """
    
    # We pass the list of tables in the prompt context (simplified)
    # In a real app, we might want to fetch schema dynamically.
    
    # Bind tools: SQL Query, Schema, and Python REPL
    llm_with_tools = model.bind_tools([run_query_tool, get_schema_tool, python_tool])
    
    # Format prompt with system message
    messages = [SystemMessage(content=prompt)] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def check_query_or_code(state: MessagesState) -> Literal["run_query", "python_node", "get_schema", END]:
    """Conditional Edge logic."""
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        tool_name = last_message.tool_calls[0]["name"]
        if tool_name == "sql_db_query":
            return "run_query"
        elif tool_name == "python_repl":
            return "python_node"
        elif tool_name == "sql_db_schema":
            return "get_schema"
            
    return END

# --- Graph Construction ---

graph_builder = StateGraph(MessagesState)

# Nodes
graph_builder.add_node("list_tables", list_tables)
graph_builder.add_node("agent", generate_query)
graph_builder.add_node("run_query", ToolNode([run_query_tool]))
graph_builder.add_node("python_node", ToolNode([python_tool]))
graph_builder.add_node("get_schema", ToolNode([get_schema_tool]))

# Edges
graph_builder.add_edge(START, "list_tables")
graph_builder.add_edge("list_tables", "agent")

# Conditional Routing from Agent
graph_builder.add_conditional_edges(
    "agent",
    check_query_or_code,
    {
        "run_query": "run_query",
        "python_node": "python_node",
        "get_schema": "get_schema",
        END: END
    }
)

# Return to Agent after tool execution to interpret results or do next step
graph_builder.add_edge("run_query", "agent")
graph_builder.add_edge("python_node", "agent")
graph_builder.add_edge("get_schema", "agent")

# Compile
graph = graph_builder.compile()

# --- Visualization (Optional) ---
try:
    from IPython.display import Image, display
    display(Image(graph.get_graph().draw_mermaid_png()))
except:
    pass

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("\n>>> SQL AGENT with VISUALIZATION <<<\n")
    
    query = "Show me the top 5 artists by number of tracks and plot a bar chart."
    print(f"User Question: {query}\n")
    
    events = graph.stream(
        {"messages": [("user", query)]},
        stream_mode="values"
    )
    
    for event in events:
        msg = event["messages"][-1]
        
        # Pretty print based on message type
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"\n[AI] Calling Tool: {msg.tool_calls[0]['name']}")
                print(f"     Args: {msg.tool_calls[0]['args']}")
            else:
                print(f"\n[AI] Response: {msg.content}")
        elif isinstance(msg, ToolMessage):
             print(f"\n[Tool] Output: {str(msg.content)[:200]}...")
