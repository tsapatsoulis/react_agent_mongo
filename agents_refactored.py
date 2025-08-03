import asyncio
from enum import Enum
import re
import subprocess
from typing import Any, Dict, TypedDict, Annotated
import uuid
from dotenv import load_dotenv
import httpx
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
import os
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import os
from typing import Callable
from dotenv import load_dotenv
from langchain_core.tools import BaseTool, tool as create_tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command
from langgraph.prebuilt.interrupt import HumanInterruptConfig, HumanInterrupt
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
# tools
from pydantic import BaseModel
from pydantic_core import Url

load_dotenv(".env.agents", override=True)

class UserFeedback(str, Enum):
    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    format_instruction: str | None

def update_first_key_copy(d: Dict, new_value: Any | str):
    if not d:
        raise ValueError("Cannot update first key of an empty dictionary")

    # Create a copy of the dictionary
    new_dict = d.copy()
    new_dict[next(iter(new_dict))] = new_value

    return new_dict

def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: HumanInterruptConfig = None,
) -> BaseTool:
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_reject": True,
        }

    @create_tool(
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )
    async def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        request: HumanInterrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": "Please review the tool call"
        }

        response = interrupt([request])[0]

        if response["type"] == UserFeedback.ACCEPT:
            tool_response = await tool.ainvoke(tool_input, config)
        elif response["type"] == UserFeedback.EDIT:
            key = list(tool.args_schema.model_fields.keys())[0]  # Get the first key from args_schema
            original_input = tool_input.copy()
            tool_input = response["args"]["args"] # this is where the new value is set
            if tool_input is not None: # or some clever validation logic here for sanitizing the input
                tool_response = await tool.ainvoke(tool_input, config)
                tool_response = f"User edited the search from '{original_input.get(key, original_input)}' to '{tool_input.get(key, tool_input)}'. You must **ONLY** answer for query '{tool_input.get(key, tool_input)}' and answer based on the edited response and **MUST NOT** reference again '{original_input.get(key, original_input)}' => {tool_response}"
            else:
                tool_response = "Tool edit failed, marking it as REJECTED!"
        elif response["type"] == UserFeedback.REJECT:
            tool_response = "Tool call REJECTED!"
        else:
            raise ValueError(f"Unsupported interrupt response type: {response['type']}")
        return tool_response
    return call_tool_with_interrupt

@create_tool
def file_reader(filepath: str) -> str:
    """Read contents of a file"""
    try:
        with open(filepath, 'r') as f:
            return f.read()[:1000]
    except Exception as e:
        return f"Error: {e}"

@create_tool
def get_files_with_extension(directory: str, extension: str) -> list[str]:
    """Get files with a specific extension in a directory"""
    try:
        return [f for f in os.listdir(directory) if f.endswith(extension)]
    except Exception as e:
        return [f"Error: {e}"]

@create_tool
def simple_math(expression: str) -> str:
    """calculate a simple math expression"""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

@create_tool
def web_search(query: str) -> str:
    """Search the web for information"""
    return f"Search results for: {query}"

@create_tool
def system_command(command: str) -> str:
    """Execute system command - DANGEROUS TOOL"""
    return subprocess.getoutput(command)

@create_tool
def fetch_ticker_info(ticker: str) -> str:
    """Fetch stock ticker information"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        return f"Information for ticker {ticker}: {info}"
    except Exception as e:
        return f"Error fetching ticker info: {e}"

class WeatherData(BaseModel):
    source_url: Url
    location: str
    condition: str
    temperature: str

@create_tool(response_format="content")
async def weather_scraper(location: str) -> WeatherData:
    """Get current weather by scraping wttr.in"""
    try:
        url = f"http://wttr.in/{location}?format=%l:+%C+%t"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5)

        text = response.text.strip()
        match = re.match(r"(.+?):\s+(.+?)\s+([\+\-]?\d+°\w+)", text)

        if match:
            return WeatherData(
                source_url=url,
                location=match.group(1),
                condition=match.group(2),
                temperature=match.group(3)
            )

        return WeatherData(source_url=url, location=location, condition="unknown", temperature="unknown")
    except:
        return WeatherData(source_url=url, location=location, condition="unavailable", temperature="unknown")

@create_tool
async def three_day_forecast(location: str) -> str:
    """Get 3-day weather forecast for a location"""
    try:
        url = f"http://wttr.in/{location}?T"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=False, timeout=5)
        return response.text.strip()
    except:
        return f"Forecast data unavailable for {location}"


TOOLS = [file_reader, web_search, system_command, get_files_with_extension, simple_math, fetch_ticker_info, add_human_in_the_loop(weather_scraper), three_day_forecast]
FORMAT_INSTRUCTION = "Format as email with subject and body."
DB_NAME = "test_react_agent"

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
    return {"messages": [await create_llm().ainvoke(state.get("messages", []))]}

async def format_node(state: AgentState) -> dict:
    if not state.get("format_instruction"):
        return {"messages": []}

    last_response = state["messages"][-1].content
    format_prompt = f"Format this response according to the instruction: '{state['format_instruction']}'\n\nOriginal response: {last_response}"

    formatted = await create_llm().ainvoke([HumanMessage(content=format_prompt)])
    return {"messages": [formatted], "format_instruction": f"Format this response according to the instruction: '{state['format_instruction']}'"}

def route_condition(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # maybe add interrupt condition here
    if state.get("format_instruction"):
        return "format"

    return END

def create_agent(checkpointer: AsyncMongoDBSaver | MemorySaver = MemorySaver()) -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))
    workflow.add_node("format", format_node)

    workflow.add_edge(START, "agent")
    workflow.add_edge("tools", "agent")
    workflow.add_edge("format", END)

    workflow.add_conditional_edges("agent", route_condition)

    return workflow.compile(checkpointer=checkpointer)

async def initialize_mongodb_saver() -> AsyncMongoDBSaver:
    mongo_uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/")
    client = AsyncIOMotorClient(mongo_uri, connect=False)
    saver = AsyncMongoDBSaver(client, db_name=DB_NAME, ttl=None) # TTL might be useful for expiring old data
    await saver._setup()

    print(f"{"=" * 20} MongoDB Saver Initialized ({DB_NAME}) {"=" * 20}")

    return saver

async def run_agent():
    saver = await initialize_mongodb_saver()
    agent = create_agent(checkpointer=saver)
    total_token_count: int = 0
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    while True:
        # Check for pending interrupts
        latest_state = await agent.aget_state(config)
        if latest_state.interrupts:
            print("\nPending Interrupts:")
            for interrupt in latest_state.interrupts:
                print(f"{interrupt.value[0].get("description")} - {interrupt.value[0]['action_request']['action']}: {interrupt.value[0]['action_request']['args']}")

            # Get user input to proceed
            user_input = input("Continue? (yes/no/edit): ").strip().lower()
            if user_input in ['quit', 'exit']:
                break
            else:
                match user_input:
                    case 'yes':
                        resume_dict = {"type": UserFeedback.ACCEPT}
                    case 'no':
                        resume_dict = {"type": UserFeedback.REJECT}
                    case 'edit':
                        edited_value = input("Please provide the edited value: ").strip()
                        updated_query = update_first_key_copy(latest_state.interrupts[0].value[0]['action_request']['args'], edited_value)
                        resume_dict = {"type": UserFeedback.EDIT, "args": {"args": updated_query}}
                    case _:
                        print("Invalid input, resuming with 'accept'.")
                        resume_dict = {"type": UserFeedback.ACCEPT}
                state = Command(resume=[resume_dict])
        else:
            user_input = input("You: ").strip().lower()
            if user_input in ['quit', 'exit']:
                break

            state = dict(messages=[HumanMessage(content=user_input)], format_instruction=FORMAT_INSTRUCTION)
        try:
            async for chunk in agent.astream(
                input=state,
                config=config,
                stream_mode="values"  # Gets the full state at each step
            ):
                if messages := chunk.get("messages"):
                    messages[-1].pretty_print()
                    if messages[-1].type == 'ai':
                        total_token_count += int(messages[-1].response_metadata['token_usage']['total_tokens'])

        except Exception as e:
            print(f"Error: {e}")
        finally:
            print(f"=> Total tokens thus far: {total_token_count}")

if __name__ == "__main__":
    asyncio.run(run_agent())