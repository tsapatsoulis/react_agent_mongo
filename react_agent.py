import os
from typing import Annotated, Dict, List, Optional, TypedDict, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
import uuid
import logging
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from pydantic import BaseModel, Field

# Configure logging for production debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env.agents", override=True)


class AgentDecision(Enum):
    """Enumeration of possible agent decision states."""

    CONTINUE = "continue"
    FINISH = "finish"
    ERROR = "error"


@dataclass
class ReasoningStep:
    """Encapsulates a single reasoning step with thought, action, and observation."""

    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ReasoningStep":
        """Reconstruct from dictionary."""
        return cls(**data)


class AgentState(TypedDict):
    """State schema for the ReAct agent workflow."""

    messages: Annotated[List, add_messages]
    reasoning_steps: List[ReasoningStep]
    current_thought: str
    iteration_count: int
    max_iterations: int
    final_answer: Optional[str]


class ReActPrompts:
    """Curated prompts for the ReAct reasoning pattern."""

    REACT_SYSTEM_PROMPT = """You are a sophisticated ReAct agent that follows the Reasoning and Acting pattern.

For each step, you must structure your response exactly as follows:

Thought: [Your reasoning about what to do next]
Action: [The action name from available tools, or "Final Answer"]
Action Input: [The input for the action as a JSON object]

Available actions:
{tool_descriptions}

Key principles:
- Always start with a "Thought" to reason about the current situation
- Use "Action" to specify what tool to use or "Final Answer" to conclude
- Provide "Action Input" as a proper JSON object
- After receiving an observation, think about what it means before proceeding
- You have a maximum of {max_iterations} iterations to solve the problem
- If no tools can satisfly the request, provide a reasoned "Final Answer"

Example format:
Thought: I need to search for information about the user's question.
Action: search_tool
Action Input: {{"query": "example search"}}

When you have enough information to answer the question:
Thought: I now have sufficient information to provide a comprehensive answer.
Action: Final Answer
Action Input: {{"answer": "Your final answer here"}}
"""

    @classmethod
    def format_system_prompt(
        cls, tools: List[BaseTool], max_iterations: int = 10
    ) -> str:
        """Formats the system prompt with available tools."""
        tool_descriptions = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in tools]
        )
        return cls.REACT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions, max_iterations=max_iterations
        )


# Example Tools for Demonstration
@tool
def search_knowledge_base(query: str) -> str:
    """Search a knowledge base for relevant information."""
    knowledge_responses = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        "react": "ReAct (Reasoning and Acting) is a paradigm that combines reasoning with action execution.",
        "machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
    }

    for keyword, response in knowledge_responses.items():
        if keyword.lower() in query.lower():
            return f"Knowledge Base Result: {response}"

    return "No relevant information found in the knowledge base."


@tool
def calculate_mathematical_expression(expression: str) -> str:
    """Safely evaluate mathematical expressions."""
    try:
        if any(char in expression for char in ["import", "exec", "eval", "__"]):
            return "Error: Invalid expression detected."

        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters."

        result = eval(expression)
        return f"Mathematical Result: {result}"
    except Exception as e:
        return f"Calculation Error: {str(e)}"


@tool
def get_current_timestamp() -> str:
    """Retrieve the current timestamp."""
    return f"Current Timestamp: {datetime.now().isoformat()}"


class ReActAgent:
    """Production-grade ReAct Agent with official MongoDB persistence."""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 10,
        mongodb_uri: str = "mongodb://localhost:27017",
        db_name: str = "langgraph_checkpoints",
        checkpoint_collection_name: str = "checkpoints",
        writes_collection_name: str = "checkpoint_writes",
        use_persistence: bool = True,
    ):
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        self.tools = tools or [
            search_knowledge_base,
            calculate_mathematical_expression,
            get_current_timestamp,
        ]
        self.max_iterations = max_iterations

        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.checkpoint_collection_name = checkpoint_collection_name
        self.writes_collection_name = writes_collection_name
        self.use_persistence = use_persistence

        self.checkpointer = None
        self.workflow = None
        self.tool_node = ToolNode(self.tools)

        if not use_persistence:
            self.checkpointer = MemorySaver()
            self.workflow = self._construct_reasoning_workflow()

    def __enter__(self):
        """Enter context manager for MongoDB checkpointer."""
        if self.use_persistence:
            self._mongodb_context = MongoDBSaver.from_conn_string(
                conn_string=self.mongodb_uri,
                db_name=self.db_name,
                checkpoint_collection_name=self.checkpoint_collection_name,
                writes_collection_name=self.writes_collection_name,
            )
            self.checkpointer = self._mongodb_context.__enter__()
            logger.info(f"Initialized MongoDB checkpointer: {self.db_name}")
        else:
            self.checkpointer = MemorySaver()

        self.workflow = self._construct_reasoning_workflow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if hasattr(self, "_mongodb_context"):
            self._mongodb_context.__exit__(exc_type, exc_val, exc_tb)

    def _construct_reasoning_workflow(self) -> StateGraph:
        """Construct workflow - only call after checkpointer is initialized."""
        if self.checkpointer is None:
            raise RuntimeError(
                "Checkpointer not initialized. Use agent in context manager."
            )

        workflow = StateGraph(AgentState)

        workflow.add_node("reasoning_engine", self._reasoning_node)
        workflow.add_node("action_executor", self.tool_node)
        workflow.add_node("reflection_synthesizer", self._reflection_node)

        workflow.add_edge(START, "reasoning_engine")
        workflow.add_conditional_edges(
            "reasoning_engine",
            self._routing_decision,
            {
                "continue": "action_executor",
                "finish": "reflection_synthesizer",
                "error": END,
            },
        )
        workflow.add_edge("action_executor", "reasoning_engine")
        workflow.add_edge("reflection_synthesizer", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def create_session_config(
        self,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict:
        """Create session configuration for MongoDB storage."""
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        config = {
            "configurable": {
                "thread_id": session_id,
            }
        }

        # NOTE: Don't set checkpoint_ns if None - LangGraph saves as empty string
        # Only set it if explicitly provided and not None/empty
        if namespace and namespace.strip():
            config["configurable"]["checkpoint_ns"] = namespace
        # If namespace is None or empty, don't include checkpoint_ns at all
        # This matches how LangGraph saves documents with empty checkpoint_ns

        if user_id:
            config["configurable"]["user_id"] = user_id

        logger.info(f"Created MongoDB config: {config}")
        return config

    def get_session_stats(self, namespace: Optional[str] = None) -> Dict:
        """Get statistics about stored sessions using MongoDB aggregation."""
        try:
            db = self.checkpointer.db
            checkpoints_collection = db[self.checkpoint_collection_name]

            # Build match filter to handle empty namespace correctly
            match_filter = {}
            if namespace:
                match_filter = {"checkpoint_ns": namespace}
            else:
                # Match documents where checkpoint_ns is empty string or doesn't exist
                match_filter = {
                    "$or": [
                        {"checkpoint_ns": ""},
                        {"checkpoint_ns": {"$exists": False}},
                    ]
                }

            pipeline = [
                {"$match": match_filter},
                {
                    "$group": {
                        "_id": "$thread_id",
                        "checkpoint_count": {"$sum": 1},
                        "latest_checkpoint": {"$max": "$checkpoint_id"},
                        "first_checkpoint": {"$min": "$checkpoint_id"},
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "total_sessions": {"$sum": 1},
                        "total_checkpoints": {"$sum": "$checkpoint_count"},
                        "avg_checkpoints_per_session": {"$avg": "$checkpoint_count"},
                    }
                },
            ]

            result = list(checkpoints_collection.aggregate(pipeline))

            if result:
                stats = result[0]
                return {
                    "total_sessions": stats.get("total_sessions", 0),
                    "total_checkpoints": stats.get("total_checkpoints", 0),
                    "avg_checkpoints_per_session": round(
                        stats.get("avg_checkpoints_per_session", 0), 2
                    ),
                    "database": self.db_name,
                    "checkpoint_collection": self.checkpoint_collection_name,
                    "writes_collection": self.writes_collection_name,
                    "namespace_filter": namespace or "empty/default",
                }
            else:
                total_docs = checkpoints_collection.count_documents({})
                return {
                    "total_sessions": 0,
                    "total_checkpoints": 0,
                    "avg_checkpoints_per_session": 0,
                    "database": self.db_name,
                    "checkpoint_collection": self.checkpoint_collection_name,
                    "writes_collection": self.writes_collection_name,
                    "namespace_filter": namespace or "empty/default",
                    "debug_total_docs_in_collection": total_docs,
                }

        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {"error": str(e)}

    def _extract_conversation_context(self, messages) -> str:
        """Extract key information from conversation history."""
        context_parts = []

        for msg in messages[-10:]:  # Look at recent messages
            if isinstance(msg, HumanMessage):
                context_parts.append(f"User asked: {msg.content}")
            elif isinstance(msg, ToolMessage):
                # Extract key information from tool results
                content = msg.content
                if "Result:" in content:
                    result_part = content.split("Result:")[-1].strip()
                    context_parts.append(f"Found: {result_part}")
                else:
                    context_parts.append(f"Tool result: {content}")

        return "\n".join(context_parts[-5:]) if context_parts else ""

    def _reasoning_node(self, state: AgentState) -> Dict:
        """Core reasoning engine that processes thoughts and determines actions."""
        if state["iteration_count"] >= state["max_iterations"]:
            return {
                "messages": [
                    AIMessage(
                        content="Maximum iterations reached. Concluding analysis."
                    )
                ],
                "final_answer": "Unable to complete task within iteration limit.",
            }

        # Get the current user query (the most recent HumanMessage)
        current_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                current_query = msg.content
                break

        # Extract conversation history and previous results from ALL messages (not just recent)
        conversation_context = self._extract_conversation_context(state["messages"])
        all_tool_results = []
        for msg in state["messages"]:  # Check ALL messages for complete context
            if isinstance(msg, ToolMessage):
                all_tool_results.append(msg.content)

        # Check if we can answer directly from conversation history
        if self._can_answer_from_history(current_query, all_tool_results):
            logger.info(
                f"🧠 Answering '{current_query}' directly from conversation history"
            )

            answer = self._synthesize_answer_from_history(
                current_query, all_tool_results
            )

            # Create a reasoning step that uses existing information
            reasoning_step = ReasoningStep(
                thought=f"I can answer this question using information already available from our conversation.",
                action="Final Answer",
                action_input={"answer": answer},
            )

            return {
                "messages": [AIMessage(content=f"Final Answer: {answer}")],
                "reasoning_steps": state["reasoning_steps"] + [reasoning_step],
                "current_thought": reasoning_step.thought,
                "iteration_count": state["iteration_count"] + 1,
                "final_answer": answer,
            }

        # If we can't answer from history, proceed with normal ReAct reasoning
        recent_tool_results = []
        for msg in state["messages"][-10:]:  # Check recent messages for context
            if isinstance(msg, ToolMessage):
                recent_tool_results.append(f"Previous result: {msg.content}")

        # Check recent actions from current reasoning session only
        recent_actions = []
        for step in state["reasoning_steps"][
            -3:
        ]:  # Only current session reasoning steps
            if hasattr(step, "action") and step.action:
                recent_actions.append(step.action)
            elif isinstance(step, dict) and step.get("action"):
                recent_actions.append(step["action"])

        # Build enhanced context prompt
        context_note = ""
        if all_tool_results:
            context_note += (
                f"\n\nPREVIOUS RESULTS FROM THIS CONVERSATION:\n"
                + "\n".join([f"- {result}" for result in all_tool_results[-5:]])
            )
            context_note += f"\n\nIMPORTANT: Before using any tools, check if the information you need is already available in the previous results above. Only use tools if you need NEW information that isn't already available."

        # Prevent repetitive actions within current reasoning session
        if len(set(recent_actions)) == 1 and len(recent_actions) >= 2:
            action_name = recent_actions[0]
            context_note += f"\n\nWARNING: You just used {action_name} multiple times in this reasoning session. Use previous results instead."

        system_prompt = (
            ReActPrompts.format_system_prompt(self.tools, self.max_iterations)
            + context_note
        )

        messages = [
            {"role": "system", "content": system_prompt},
            *[self._message_to_dict(msg) for msg in state["messages"]],
        ]

        response = self.llm.invoke(messages)
        parsed_response = self._parse_react_response(response.content)

        # Create reasoning step
        reasoning_step = ReasoningStep(
            thought=parsed_response.get("thought", ""),
            action=parsed_response.get("action"),
            action_input=parsed_response.get("action_input"),
        )

        updated_state = {
            "messages": [response],
            "reasoning_steps": state["reasoning_steps"] + [reasoning_step],
            "current_thought": reasoning_step.thought,
            "iteration_count": state["iteration_count"] + 1,
        }

        # Handle Final Answer
        if reasoning_step.action == "Final Answer":
            final_answer = (
                reasoning_step.action_input.get("answer", response.content)
                if reasoning_step.action_input
                else response.content
            )
            updated_state["final_answer"] = final_answer

        return updated_state

    def _can_answer_from_history(self, query: str, all_tool_results: List[str]) -> bool:
        """Simple pattern-based check for obvious continuation queries."""
        if not all_tool_results:
            return False

        query_lower = query.lower()

        # Only catch the most obvious continuation patterns
        obvious_continuations = [
            "continue our conversation",
            "what was the result",
            "what did we just",
            "remind me of",
        ]

        return any(pattern in query_lower for pattern in obvious_continuations)

    def _synthesize_answer_from_history(
        self, query: str, all_tool_results: List[str]
    ) -> str:
        """Generic answer synthesis from conversation history."""

        # Simple but flexible approach
        if not all_tool_results:
            return "No previous information available."

        # For continuation queries, return the most recent relevant result
        query_lower = query.lower()
        if any(
            phrase in query_lower
            for phrase in ["continue", "what was", "remind me", "result"]
        ):
            return f"Based on our previous conversation: {all_tool_results[-1]}"

        # Otherwise, let the ReAct agent handle it normally
        return (
            f"I have previous information available: {' '.join(all_tool_results[-2:])}"
        )

    def _reflection_node(self, state: AgentState) -> Dict:
        """Synthesizes the reasoning journey into a coherent final response."""
        final_response = state.get("final_answer", "Task completed.")

        # SAFETY CHECK: Ensure we always have a meaningful response
        if not final_response or final_response.strip() == "":
            logger.warning("⚠️ Empty final_answer detected, creating fallback response")

            # Try to extract answer from the last AI message
            last_ai_message = None
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg.content
                    break

            if last_ai_message:
                final_response = last_ai_message
            else:
                final_response = "I processed your request but encountered an issue generating the final response. Please try rephrasing your question."

        reasoning_summary = self._synthesize_reasoning_journey(state["reasoning_steps"])

        reflection_message = AIMessage(
            content=f"Final Answer: {final_response}\n\nReasoning Summary:\n{reasoning_summary}"
        )

        logger.info(
            f"✅ Reflection node sending final response: {final_response[:100]}..."
        )
        return {"messages": [reflection_message]}

    def _routing_decision(self, state: AgentState) -> str:
        """Determines the next node in the reasoning workflow."""
        if state.get("final_answer"):
            return "finish"

        if state["iteration_count"] >= state["max_iterations"]:
            return "error"

        last_reasoning_step = (
            state["reasoning_steps"][-1] if state["reasoning_steps"] else None
        )

        if (
            last_reasoning_step
            and hasattr(last_reasoning_step, "action")
            and last_reasoning_step.action
            and last_reasoning_step.action != "Final Answer"
        ):
            return "continue"
        elif (
            last_reasoning_step
            and isinstance(last_reasoning_step, dict)
            and last_reasoning_step.get("action")
            and last_reasoning_step.get("action") != "Final Answer"
        ):
            return "continue"

        return "finish"

    def _parse_react_response(self, response_content: str) -> Dict:
        """Elegantly parses ReAct formatted responses."""
        patterns = {
            "thought": r"Thought:\s*(.*?)(?=\n(?:Action|$))",
            "action": r"Action:\s*(.*?)(?=\n(?:Action Input|$))",
            "action_input": r"Action Input:\s*(.*?)(?=\n(?:Thought|Action|$)|$)",
        }

        parsed = {}

        for key, pattern in patterns.items():
            match = re.search(pattern, response_content, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if key == "action_input":
                    try:
                        parsed[key] = json.loads(content)
                    except json.JSONDecodeError:
                        parsed[key] = {"input": content}
                else:
                    parsed[key] = content

        return parsed

    def _synthesize_reasoning_journey(
        self, reasoning_steps: List[ReasoningStep]
    ) -> str:
        """Creates an elegant summary of the reasoning process."""
        if not reasoning_steps:
            return "No reasoning steps recorded."

        summary_lines = []
        for i, step in enumerate(reasoning_steps, 1):
            if hasattr(step, "thought"):
                summary_lines.append(f"Step {i}: {step.thought}")
                if hasattr(step, "action") and step.action:
                    summary_lines.append(f"  → Action: {step.action}")
                if hasattr(step, "observation") and step.observation:
                    summary_lines.append(f"  → Observation: {step.observation}")
            elif isinstance(step, dict):
                summary_lines.append(
                    f"Step {i}: {step.get('thought', 'No thought recorded')}"
                )
                if step.get("action"):
                    summary_lines.append(f"  → Action: {step['action']}")
                if step.get("observation"):
                    summary_lines.append(f"  → Observation: {step['observation']}")

        return "\n".join(summary_lines)

    def _message_to_dict(self, message) -> Dict:
        """Converts LangChain messages to dictionary format."""
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, ToolMessage):
            return {"role": "tool", "content": message.content}
        else:
            return {"role": "system", "content": str(message.content)}

    def invoke(
        self,
        query: str,
        config: Optional[Dict] = None,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> Dict:
        """Synchronously process queries with MongoDB session management."""
        if config is None:
            config = self.create_session_config(session_id, namespace)

        logger.info(f"Processing query with config: {config}")

        try:
            # Debug the get_tuple call
            logger.info(f"Checking for cached state with config: {config}")
            cached_state = self.checkpointer.get_tuple(config)
            logger.info(f"Cached state result: {cached_state is not None}")

            if cached_state:
                logger.info(
                    f"✅ RESUMING from cached state for session: {config['configurable']['thread_id']}"
                )
                logger.info(
                    f"Cached checkpoint ID: {cached_state.checkpoint.get('id', 'unknown')}"
                )
                existing_messages = cached_state.checkpoint.get(
                    "channel_values", {}
                ).get("messages", [])
                logger.info(f"Found {len(existing_messages)} existing messages")

                # CRITICAL FIX: Reset reasoning_steps for each new query
                # Only preserve conversation history (messages), not the reasoning steps
                initial_state = {
                    "messages": existing_messages + [HumanMessage(content=query)],
                    "reasoning_steps": [],  # Reset reasoning steps for new query
                    "current_thought": "",  # Reset current thought
                    "iteration_count": 0,  # Reset iteration count
                    "max_iterations": self.max_iterations,
                    "final_answer": None,  # Reset final answer
                }
            else:
                logger.info(
                    f"🆕 STARTING fresh session: {config['configurable']['thread_id']}"
                )

                # Check what's actually in the database
                try:
                    db = self.checkpointer.db
                    checkpoints_collection = db[self.checkpoint_collection_name]

                    # Try different filter combinations to see what exists
                    thread_filter = {"thread_id": config["configurable"]["thread_id"]}

                    thread_count = checkpoints_collection.count_documents(thread_filter)
                    total_count = checkpoints_collection.count_documents({})

                    logger.info(
                        f"📊 DB Debug - Total docs: {total_count}, Thread docs: {thread_count}"
                    )

                    # Show sample documents to understand the structure
                    sample_docs = list(
                        checkpoints_collection.find(thread_filter).limit(2)
                    )
                    for doc in sample_docs:
                        logger.info(
                            f"📄 Sample doc: thread_id={doc.get('thread_id')}, checkpoint_ns={doc.get('checkpoint_ns')}"
                        )

                except Exception as db_error:
                    logger.error(f"DB debug error: {db_error}")

                initial_state = {
                    "messages": [HumanMessage(content=query)],
                    "reasoning_steps": [],
                    "current_thought": "",
                    "iteration_count": 0,
                    "max_iterations": self.max_iterations,
                    "final_answer": None,
                }
        except Exception as e:
            logger.error(f"Error getting cached state: {e}")
            import traceback

            traceback.print_exc()
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "reasoning_steps": [],
                "current_thought": "",
                "iteration_count": 0,
                "max_iterations": self.max_iterations,
                "final_answer": None,
            }

        try:
            result = self.workflow.invoke(initial_state, config=config)

            # CRITICAL SAFETY CHECK: Ensure user always gets a response
            final_messages = result.get("messages", [])
            if not final_messages:
                logger.error("❌ No messages in result - adding emergency response")
                result["messages"] = [
                    AIMessage(
                        content="I processed your request but didn't generate a proper response. Please try again."
                    )
                ]

            # Check if the last message is meaningful
            last_message = final_messages[-1] if final_messages else None
            if not last_message or not last_message.content.strip():
                logger.error("❌ Empty final message - adding emergency response")
                if not final_messages:
                    result["messages"] = []
                result["messages"].append(
                    AIMessage(
                        content="I encountered an issue generating a response. Based on our conversation, please let me know if you need clarification on anything."
                    )
                )

            session_info = {
                "session_id": config["configurable"].get("thread_id"),
                "namespace": config["configurable"].get("checkpoint_ns", "default"),
                "timestamp": datetime.now().isoformat(),
            }

            result["session_info"] = session_info
            return result

        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR in workflow execution: {e}")
            import traceback

            traceback.print_exc()

            # EMERGENCY RESPONSE: Always provide something to the user
            emergency_result = {
                "messages": [
                    AIMessage(
                        content=f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question or contact support if the issue persists."
                    )
                ],
                "session_info": {
                    "session_id": config["configurable"].get("thread_id"),
                    "namespace": config["configurable"].get("checkpoint_ns", "default"),
                    "timestamp": datetime.now().isoformat(),
                    "error": True,
                },
            }
            return emergency_result


def demonstrate_react_agent():
    """Demonstrates ReAct agent with proper MongoDB context management."""
    with ReActAgent(
        llm=ChatOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("OPENAI_KEY"),
            model="openai/gpt-4.1",
            temperature=0.8,
        ),
        max_iterations=8,
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_prod",
        use_persistence=True,
    ) as agent:

        print("✅ Initialized ReAct Agent with MongoDB persistence")

        scenarios = [
            {
                "query": "What is Python and calculate 15 * 23?",
                "session_id": "prod_session_001",
                "namespace": None,  # Don't use namespace to match existing data
            },
            {
                "query": "Continue our conversation - what was the result?",
                "session_id": "prod_session_001",
                "namespace": None,  # Don't use namespace to match existing data
            },
        ]

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n🔍 Test {i}: {scenario['query']}")

            try:
                result = agent.invoke(
                    query=scenario["query"],
                    session_id=scenario["session_id"],
                    namespace=scenario["namespace"],
                )

                final_messages = result.get("messages", [])
                if final_messages:
                    print(f"📋 Response: {final_messages[-1].content}")

                # Show session stats
                stats = agent.get_session_stats(namespace=scenario["namespace"])
                print(f"📊 Session Stats: {stats}")

            except Exception as e:
                print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    demonstrate_react_agent()
