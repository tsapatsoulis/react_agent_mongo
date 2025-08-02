import asyncio
import subprocess
from typing import TypedDict, Annotated, Literal
from dataclasses import dataclass
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
import os
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv(".env.agents", override=True)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pending_dangerous_call: dict | None
    awaiting_permission: bool
    format_instruction: str | None


@tool
def file_reader(filepath: str) -> str:
    """Read contents of a file"""
    try:
        with open(filepath, 'r') as f:
            return f.read()[:1000]
    except Exception as e:
        return f"Error: {e}"

@tool
def get_files_with_extension(directory: str, extension: str) -> list[str]:
    """Get files with a specific extension in a directory"""
    try:
        return [f for f in os.listdir(directory) if f.endswith(extension)]
    except Exception as e:
        return [f"Error: {e}"]

@tool
def simple_math(expression: str) -> str:
    """calculate a simple math expression"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

@tool
def web_search(query: str) -> str:
    """Search the web for information"""
    return f"Search results for: {query}"

@tool
def system_command(command: str) -> str:
    """Execute system command - DANGEROUS TOOL"""
    return subprocess.getoutput(command)

TOOLS = [file_reader, web_search, system_command, get_files_with_extension, simple_math]
DANGEROUS_TOOLS = {"system_command", "simple_math"}

def create_llm():
    return AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_AZURE_API_KEY"),
            deployment_name=os.getenv("CHAT_MODELS_DEPLOYMENT_NAME"),
            model_name=os.getenv("CHAT_MODEL_NAME"),
            azure_endpoint=os.getenv("OPENAI_AZURE_API_BASE"),
            api_version=os.getenv("OPENAI_AZURE_API_VERSION"),
            temperature=0.3,
        ).bind_tools(TOOLS, tool_choice="auto")

async def agent_node(state: AgentState) -> dict:
    if state.get("awaiting_permission"):
        last_msg = state["messages"][-1].content.lower().strip()
        dangerous_call = state["pending_dangerous_call"]

        if last_msg.startswith("no"):
            return {
                "messages": [AIMessage(content="Operation cancelled")],
                "pending_dangerous_call": None,
                "awaiting_permission": False,
                "format_instruction": "Explain the user that the operation was cancelled.",
            }
        elif last_msg.startswith("yes"):
            return {
                "messages": [AIMessage(content="", tool_calls=[dangerous_call])],
                "pending_dangerous_call": None,
                "awaiting_permission": False,
            }
        else:
            key = list(dangerous_call["args"].keys())[0]
            dangerous_call["args"] = {key: last_msg}

            state["messages"] = [HumanMessage(content=last_msg)]

            return {
                "messages": [HumanMessage(content=last_msg), AIMessage(content="", tool_calls=[dangerous_call])],
                "pending_dangerous_call": None,
                "awaiting_permission": False,
            }

    messages_for_llm = []
    for msg in state["messages"]:
        if not (isinstance(msg, AIMessage) and "⚠️ DANGEROUS:" in msg.content):
            messages_for_llm.append(msg)

    llm = create_llm()
    response = await llm.ainvoke(messages_for_llm)

    if response.tool_calls:
        for call in response.tool_calls:
            if call["name"] in DANGEROUS_TOOLS:
                args_str = ", ".join(f"{k}: {v}" for k, v in call["args"].items())
                return {
                    "messages": [AIMessage(content=f"⚠️ DANGEROUS: {call['name']}")],
                    "pending_dangerous_call": call,
                    "awaiting_permission": True,
                }

    return {"messages": [response]}

async def format_node(state: AgentState) -> dict:
    if not state.get("format_instruction"):
        return {"messages": []}

    last_response = state["messages"][-1].content
    format_prompt = f"Format this response according to the instruction: '{state['format_instruction']}'\n\nOriginal response: {last_response}"

    llm = create_llm()
    formatted = await llm.ainvoke([HumanMessage(content=format_prompt)])
    return {"messages": [formatted], "format_instruction": f"Format this response according to the instruction: '{state['format_instruction']}'"}


def route_condition(state: AgentState) -> Literal["agent", "tools", "format", "__end__"]:
    if state.get("awaiting_permission"):
        return END

    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    if state.get("format_instruction") and not ("⚠️" in last_message.content):
        return "format"

    return END

async def create_agent(checkpointer: AsyncMongoDBSaver | MemorySaver = MemorySaver()) -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("format", format_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_condition)
    workflow.add_edge("tools", "agent")
    workflow.add_edge("format", END)

    return workflow.compile(checkpointer=checkpointer)

async def run_agent():

    mongo_uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000")
    client = AsyncIOMotorClient(mongo_uri)
    saver = AsyncMongoDBSaver(client)
    await saver._setup()

    agent = await create_agent(checkpointer=saver)
    config = {"configurable": {"thread_id": "test-thread"}}
    format_instruction = "Format as email with subject and body."

    try:
        state: StateGraph = await agent.aget_state(config)
        if state.values:
            print(f"Loaded checkpoint with {len(state.values.get('messages', []))} messages")
            if state.values.get("awaiting_permission") and state.values.get("pending_dangerous_call"):
                last_msg = [msg for msg in state.values["messages"] if not ("⚠️ DANGEROUS:" in getattr(msg, 'content', ''))][-1]
                dangerous_call = state.values["pending_dangerous_call"]
                args_str = ", ".join(f"{k}: {v}" for k, v in dangerous_call["args"].items())

                print(f"Last message: {last_msg.content}")
                print(f"Interrupted tool: {dangerous_call['name']} with args: {args_str}")

            state = state.values
        else:
            print("No checkpoint found, starting fresh")
            state = {"messages": [], "pending_dangerous_call": None, "awaiting_permission": False, "format_instruction": format_instruction}
    except:
        print("Starting fresh")
        state = {"messages": [], "pending_dangerous_call": None, "awaiting_permission": False, "format_instruction": format_instruction}

    print("Agent ready! Type 'quit' to exit.")

    while True:
        if not state.get("awaiting_permission"):
            user_input = input("User: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            state["messages"].append(HumanMessage(content=user_input))
        else:
            user_input = input("Proceed? (yes/no) or rewrite: ")
            state["messages"].append(HumanMessage(content=user_input))

        try:
            async for chunk in agent.astream(
                input=state,
                config=config,
                stream_mode="updates"
            ):
                for node_name, node_output in chunk.items():
                    if "messages" in node_output:
                        for message in node_output["messages"]:
                            if hasattr(message, 'content') and message.content:
                                message.pretty_print()
                        state.update({k: node_output.get(k, state.get(k)) for k in ["pending_dangerous_call", "awaiting_permission"]})

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_agent())