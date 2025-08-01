import asyncio
import os
from typing import Annotated, Dict, List, Optional, TypedDict, Union, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from urllib.parse import parse_qs, urlparse
import uuid
import logging
from datetime import datetime
from contextlib import contextmanager

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool, tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools import YouTubeSearchTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env.agents", override=True)


class InterruptDecision(str, Enum):
    """Interrupt decision states with string inheritance for seamless serialization."""

    NONE = "none"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDIT_REQUESTED = "edit_requested"


class AgentState(TypedDict):
    """Sophisticated state schema with interrupt-aware architecture."""

    messages: Annotated[List[BaseMessage], add_messages]
    reasoning_steps: List[Dict[str, Any]]
    current_thought: str
    iteration_count: int
    max_iterations: int
    final_answer: Optional[str]
    interrupt_decision: str
    pending_tool_call: Optional[Dict[str, Any]]
    interrupt_message: Optional[str]
    awaiting_tool_approval: bool
    tool_call_mapping: Dict[str, str]
    rejected_tools: List[str]
    rejection_reasons: Dict[str, str]
    approved_tools: List[str]  # Session-scoped approval memory
    edited_tool_input: Optional[Dict[str, Any]]
    output_formatting_prompt: Optional[str]  # User-defined formatting directive
    raw_final_answer: Optional[str]  # Pres


@dataclass
class ToolSafetyProfile:
    """Encapsulates tool safety configuration with semantic clarity."""

    tool_name: str
    requires_approval: bool
    risk_level: Literal["low", "medium", "high", "critical"]
    approval_prompt: str
    risk_rationale: str
    allows_input_editing: bool = True

    def format_approval_request(self, tool_input: Dict[str, Any]) -> str:
        """Generate contextually rich approval request with input editing option."""
        input_display = json.dumps(tool_input, indent=2)

        base_request = (
            f"## 🔐 Human Approval Required\n\n"
            f"**Tool:** `{self.tool_name}`\n"
            f"**Risk Level:** {self.risk_level.upper()}\n"
            f"**Rationale:** {self.risk_rationale}\n\n"
            f"### Requested Operation\n"
            f"```json\n{input_display}\n```\n\n"
            f"{self.approval_prompt}\n\n"
            f"**📌 Note:** Approving will authorize all future uses of `{self.tool_name}` in this session.\n\n"
        )

        if self.allows_input_editing:
            base_request += (
                f"**Options:**\n"
                f"- **Approve**: Execute with current input\n"
                f"- **Edit**: Modify the input before execution\n"
                f"- **Reject**: Cancel this operation\n\n"
                f"**Please choose: approve, edit, or reject**"
            )
        else:
            base_request += "**Please approve or reject this operation.**"

        return base_request


class ToolSafetyRegistry:
    """Centralized tool safety management with extensible architecture."""

    def __init__(self):
        self._profiles: Dict[str, ToolSafetyProfile] = {}
        self._initialize_safety_profiles()

    def _initialize_safety_profiles(self):
        """Configure comprehensive safety profiles with semantic precision."""
        self.register_profile(
            ToolSafetyProfile(
                tool_name="calculate_mathematical_expression",
                requires_approval=True,
                risk_level="medium",
                approval_prompt="This will execute a mathematical calculation using Python's eval function.",
                risk_rationale="Potential code execution vulnerability through eval()",
                allows_input_editing=True,
            )
        )

        self.register_profile(
            ToolSafetyProfile(
                tool_name="search_knowledge_base",
                requires_approval=True,
                risk_level="low",
                approval_prompt="",
                risk_rationale="Read-only operation on curated knowledge base",
            )
        )

        self.register_profile(
            ToolSafetyProfile(
                tool_name="get_current_timestamp",
                requires_approval=False,
                risk_level="low",
                approval_prompt="",
                risk_rationale="System time access with no side effects",
            )
        )
        self.register_profile(
            ToolSafetyProfile(
                tool_name="search_wikipedia",
                requires_approval=False,  # Generally safe for information retrieval
                risk_level="low",
                approval_prompt="",
                risk_rationale="Read-only Wikipedia access with no side effects",
            )
        )
        self.register_profile(
            ToolSafetyProfile(
                tool_name="search_web_duckduckgo",
                requires_approval=True,
                risk_level="high",
                approval_prompt="This will perform web search using DuckDuckGo.",
                risk_rationale="Privacy-focused web search with no data collection",
            )
        )
        self.register_profile(
            ToolSafetyProfile(
                tool_name="search_financial_news",
                requires_approval=False,
                risk_level="low",
                approval_prompt="",
                risk_rationale="Read-only financial news retrieval with no transactional exposure",
                allows_input_editing=True,
            )
        )

        self.register_profile(
            ToolSafetyProfile(
                tool_name="analyze_market_sentiment",
                requires_approval=False,
                risk_level="low",
                approval_prompt="",
                risk_rationale="Comprehensive market analysis without financial transaction capabilities",
                allows_input_editing=True,
            )
        )
        self.register_profile(
            ToolSafetyProfile(
                tool_name="orchestrate_youtube_intelligence",
                requires_approval=False,
                risk_level="low",
                approval_prompt="",
                risk_rationale="Read-only content discovery through YouTube's public interface",
                allows_input_editing=True,
            )
        )

        self.register_profile(
            ToolSafetyProfile(
                tool_name="discover_educational_content",
                requires_approval=False,
                risk_level="low",
                approval_prompt="",
                risk_rationale="Pedagogical content curation with educational value optimization",
                allows_input_editing=True,
            )
        )

    def register_profile(self, profile: ToolSafetyProfile):
        """Register tool safety profile with validation."""
        self._profiles[profile.tool_name] = profile

    def requires_approval(self, tool_name: str) -> bool:
        """Determine approval requirement with fail-safe default."""
        profile = self._profiles.get(tool_name)
        return profile.requires_approval if profile else True

    def generate_approval_request(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> str:
        """Generate formatted approval request with graceful fallback."""
        profile = self._profiles.get(tool_name)

        if not profile:
            return (
                f"## ⚠️ Unregistered Tool Approval Required\n\n"
                f"**Tool:** `{tool_name}` (UNREGISTERED)\n"
                f"**Input:** ```json\n{json.dumps(tool_input, indent=2)}\n```\n\n"
                f"This tool is not in the safety registry. Proceed with extreme caution."
            )

        return profile.format_approval_request(tool_input)


class ReActPrompts:
    """Sophisticated prompt engineering with semantic precision."""

    SYSTEM_TEMPLATE = """You are an advanced ReAct agent implementing sophisticated reasoning and action patterns.

Your responses must follow this exact structure:

Thought: [Analytical reasoning about the current situation and next steps]
Action: [Tool name from available tools, or "Final Answer"]
Action Input: [Valid JSON object for the tool]

Available tools:
{tool_descriptions}

Operational constraints:
- Maximum iterations: {max_iterations}
- Always begin with analytical thought
- Provide well-formed JSON for Action Input
- Use "Final Answer" action to conclude

Example interaction:
Thought: The user needs mathematical computation. I'll calculate using the appropriate tool.
Action: calculate_mathematical_expression
Action Input: {{"expression": "42 * 17"}}

For conclusions:
Thought: I have gathered sufficient information to provide a comprehensive response.
Action: Final Answer
Action Input: {{"answer": "Based on my analysis, [detailed response]"}}
"""

    @classmethod
    def format_system_prompt(
        cls, tools: List[BaseTool], max_iterations: int = 10
    ) -> str:
        """Generate formatted system prompt with semantic clarity."""
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in tools
        )

        return cls.SYSTEM_TEMPLATE.format(
            tool_descriptions=tool_descriptions, max_iterations=max_iterations
        )


class MessageSerializer:
    """Elegant message serialization with proper type handling."""

    @staticmethod
    def serialize_for_llm(message: BaseMessage) -> Dict[str, str]:
        """Convert BaseMessage to LLM-compatible format with type safety."""
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        elif isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, ToolMessage):
            # For LLM context, represent tool results as system messages
            return {
                "role": "system",
                "content": f"Tool Result ({message.name or 'unknown'}): {message.content}",
            }
        else:
            return {"role": "system", "content": str(message.content)}

    @staticmethod
    def deserialize_from_checkpoint(messages: List[Any]) -> List[BaseMessage]:
        """Reconstruct BaseMessage objects from checkpoint data with robustness."""
        reconstructed = []

        for msg in messages:
            try:
                if isinstance(msg, BaseMessage):
                    reconstructed.append(msg)
                elif isinstance(msg, dict):
                    # Handle dictionary representations
                    msg_type = msg.get("type", "").lower()
                    content = msg.get("content", "")

                    if msg_type == "human":
                        reconstructed.append(HumanMessage(content=content))
                    elif msg_type == "ai":
                        reconstructed.append(AIMessage(content=content))
                    elif msg_type == "tool":
                        # Ensure tool_call_id exists
                        tool_call_id = msg.get("tool_call_id", str(uuid.uuid4()))
                        reconstructed.append(
                            ToolMessage(
                                content=content,
                                tool_call_id=tool_call_id,
                                name=msg.get("name", "unknown"),
                            )
                        )
                    else:
                        reconstructed.append(SystemMessage(content=content))
                else:
                    # Fallback for unknown types
                    reconstructed.append(SystemMessage(content=str(msg)))

            except Exception as e:
                logger.warning(f"Message deserialization warning: {e}")
                reconstructed.append(SystemMessage(content=str(msg)))

        return reconstructed


class AsyncReActAgent:
    """Production-grade ReAct agent with sophisticated interrupt mechanics."""

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 10,
        mongodb_uri: str = "mongodb://localhost:27017",
        db_name: str = "langgraph_checkpoints",
        checkpoint_collection_name: str = "agent_checkpoints",
        writes_collection_name: str = "agent_checkpoint_writes",
        use_persistence: bool = True,
        enable_human_interrupts: bool = True,
        default_output_formatter: Optional[str] = None,
    ):
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        self.tools = tools or self._default_tools()
        self.max_iterations = max_iterations
        self.enable_human_interrupts = enable_human_interrupts
        self.tool_safety_registry = ToolSafetyRegistry()
        self.message_serializer = MessageSerializer()

        # MongoDB configuration
        self.mongodb_uri = mongodb_uri
        self.db_name = db_name
        self.checkpoint_collection_name = checkpoint_collection_name
        self.writes_collection_name = writes_collection_name
        self.use_persistence = use_persistence

        # Initialize components
        self.checkpointer = None
        self.workflow = None
        self._tool_executor_map = {tool.name: tool for tool in self.tools}
        self.default_output_formatter = default_output_formatter

        if not use_persistence:
            self.checkpointer = MemorySaver()
            self.workflow = self._construct_workflow()

    @staticmethod
    def _default_tools() -> List[BaseTool]:
        """Provide default tool suite with semantic clarity."""
        return [
            search_knowledge_base,
            calculate_mathematical_expression,
            get_current_timestamp,
            search_wikipedia,
            # search_web_duckduckgo,
            search_financial_news,
            analyze_market_sentiment,
            orchestrate_youtube_intelligence
        ]

    async def __aenter__(self):
        """Async context manager entry with elegant resource management."""
        if self.use_persistence:
            # Use AsyncMongoDBSaver instead of MongoDBSaver
            self._mongodb_context = AsyncMongoDBSaver.from_conn_string(
                conn_string=self.mongodb_uri,
                db_name=self.db_name,
                checkpoint_collection_name=self.checkpoint_collection_name,
                writes_collection_name=self.writes_collection_name,
            )
            self.checkpointer = await self._mongodb_context.__aenter__()
            logger.info(f"Initialized async MongoDB checkpointer: {self.db_name}")
        else:
            # InMemorySaver supports async operations natively
            self.checkpointer = MemorySaver()

        self.workflow = self._construct_workflow()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper resource cleanup."""
        if hasattr(self, "_mongodb_context"):
            await self._mongodb_context.__aexit__(exc_type, exc_val, exc_tb)

    def _construct_workflow(self) -> StateGraph:
        """Construct interrupt-aware workflow with architectural elegance."""
        if not self.checkpointer:
            raise RuntimeError(
                "Checkpointer must be initialized before workflow construction"
            )

        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("reasoning_engine", self._reasoning_node)
        workflow.add_node("tool_selector", self._tool_selector_node)
        workflow.add_node("approval_checkpoint", self._approval_checkpoint_node)
        workflow.add_node("tool_executor", self._tool_executor_node)
        workflow.add_node("synthesizer", self._synthesis_node)
        workflow.add_node("handle_rejection", self._handle_rejection_node)
        workflow.add_node(
            "continuation_orchestrator", self._continuation_orchestrator_node
        )

        # Entry point
        workflow.add_edge(START, "reasoning_engine")

        # Reasoning routes
        workflow.add_conditional_edges(
            "reasoning_engine",
            self._route_from_reasoning,
            {
                "select_tool": "tool_selector",
                "synthesize": "synthesizer",
                "continue": "continuation_orchestrator",
            },
        )

        # Tool selection routes
        workflow.add_conditional_edges(
            "tool_selector",
            self._route_from_selector,
            {
                "needs_approval": "approval_checkpoint",
                "execute_directly": "tool_executor",
            },
        )

        # Approval checkpoint routes
        workflow.add_conditional_edges(
            "approval_checkpoint",
            self._route_from_approval,
            {
                "await_approval": END,
                "proceed": "tool_executor",
                "reject": "handle_rejection",
            },
        )

        # Tool executor ALWAYS goes to continuation orchestrator
        workflow.add_edge("tool_executor", "continuation_orchestrator")

        # Continuation orchestrator decides next action
        workflow.add_conditional_edges(
            "continuation_orchestrator",
            self._route_from_continuation,
            {
                "continue_reasoning": "reasoning_engine",
                "complete": "synthesizer",
            },
        )

        workflow.add_edge("synthesizer", END)
        workflow.add_edge("handle_rejection", "reasoning_engine")

        # Interrupt configuration
        interrupt_nodes = (
            ["approval_checkpoint"] if self.enable_human_interrupts else []
        )

        return workflow.compile(
            checkpointer=self.checkpointer, interrupt_before=interrupt_nodes
        )

    async def _continuation_orchestrator_node(
        self, state: AgentState
    ) -> Dict[str, Any]:
        """
        Orchestrate workflow continuation with sophisticated multi-tool awareness.

        This node serves as the cerebral cortex of the workflow, maintaining
        contextual awareness across tool executions and ensuring task completion.
        """
        # Extract conversation context
        original_query = None
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                original_query = msg.content
                break

        if not original_query:
            return {"continue_to_synthesis": True}

        # Analyze execution trajectory
        executed_tools = []
        tool_results = {}

        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                executed_tools.append(msg.name)
                tool_results[msg.name] = msg.content

        # Extract pending tasks from AI reasoning
        ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        last_ai_reasoning = ai_messages[-1].content if ai_messages else ""

        # Sophisticated task completion analysis
        continuation_prompt = f"""
        Analyze the workflow state and determine next action.

        Original request: {original_query}

        Tools executed: {executed_tools}
        Results obtained: {json.dumps(tool_results, indent=2)}

        Last AI reasoning: {last_ai_reasoning}

        The user requested multiple calculations and information retrieval.
        Determine if ALL requested tasks have been completed.

        Response format:
        {{
            "tasks_completed": ["list of completed tasks"],
            "tasks_remaining": ["list of remaining tasks"],
            "all_complete": true/false,
            "next_action": "continue" or "synthesize"
        }}
        """

        try:
            analysis_messages = [
                {
                    "role": "system",
                    "content": "You are a task completion analyzer. Be precise about what has and hasn't been done.",
                },
                {"role": "user", "content": continuation_prompt},
            ]

            response = await self.llm.ainvoke(analysis_messages)

            # Parse with fallback logic
            try:
                analysis = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback: Continue if we haven't seen a "Final Answer" action
                has_final_answer = any(
                    "Final Answer" in step.get("action", "")
                    for step in state.get("reasoning_steps", [])
                )

                if (
                    not has_final_answer
                    and state["iteration_count"] < state["max_iterations"]
                ):
                    return self._preserve_critical_state(
                        state, {"continue_to_synthesis": False}
                    )
                else:
                    return self._preserve_critical_state(
                        state, {"continue_to_synthesis": True}
                    )

            # Make decision based on analysis
            if (
                analysis.get("all_complete", False)
                or analysis.get("next_action") == "synthesize"
            ):
                return self._preserve_critical_state(
                    state, {"continue_to_synthesis": True}
                )
            else:
                # Inject continuation guidance
                remaining_tasks = analysis.get("tasks_remaining", [])
                if remaining_tasks:
                    continuation_message = SystemMessage(
                        content=f"Continue with remaining tasks: {', '.join(remaining_tasks)}"
                    )
                    return self._preserve_critical_state(
                        state,
                        {
                            "messages": [continuation_message],
                            "continue_to_synthesis": False,
                        },
                    )
                else:
                    return self._preserve_critical_state(
                        state, {"continue_to_synthesis": False}
                    )

        except Exception as e:
            logger.warning(f"Continuation analysis failed: {e}, continuing execution")
            # Conservative fallback: continue if under iteration limit
            if state["iteration_count"] < state["max_iterations"] - 1:
                return self._preserve_critical_state(
                    state, {"continue_to_synthesis": False}
                )
            else:
                return self._preserve_critical_state(
                    state, {"continue_to_synthesis": True}
                )

    async def _reasoning_node(self, state: AgentState) -> Dict[str, Any]:
        """Core reasoning engine with sophisticated state management and approval awareness."""

        # Handle rejection flow
        if state.get("interrupt_decision") == InterruptDecision.REJECTED.value:
            pending_tool = state.get("pending_tool_call", {})
            rejected_tool = pending_tool.get("tool_name")

            rejected_tools = state.get("rejected_tools", [])
            rejection_reasons = state.get("rejection_reasons", {})

            if rejected_tool and rejected_tool not in rejected_tools:
                rejected_tools.append(rejected_tool)
                rejection_reasons[rejected_tool] = (
                    "User rejected during approval process"
                )

            return self._preserve_critical_state(
                state,
                {
                    "messages": [
                        AIMessage(
                            content=f"Understanding rejection of {rejected_tool}. Continuing with alternative approach."
                        )
                    ],
                    "interrupt_decision": InterruptDecision.NONE.value,
                    "pending_tool_call": None,
                    "awaiting_tool_approval": False,
                    "rejected_tools": rejected_tools,
                    "rejection_reasons": rejection_reasons,
                    "approved_tools": state.get("approved_tools", []),  # Preserve
                },
            )

        # Check iteration limits
        if state["iteration_count"] >= state["max_iterations"]:
            return self._preserve_critical_state(
                state,
                {
                    "messages": [AIMessage(content="Maximum iterations reached.")],
                    "final_answer": "I've reached the maximum number of reasoning steps.",
                    "approved_tools": state.get("approved_tools", []),  # Preserve
                },
            )

        # Filter messages for AI context
        filtered_messages = []
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                if msg.additional_kwargs.get("requires_approval"):
                    continue
                if msg.additional_kwargs.get("synthesized"):
                    continue
            filtered_messages.append(msg)

        # Build enhanced system prompt
        system_prompt = self._build_enhanced_system_prompt(state)

        # Add approved tools context
        approved_tools = state.get("approved_tools", [])
        if approved_tools:
            system_prompt += f"\n\nPRE-APPROVED TOOLS: {', '.join(approved_tools)} - These tools can be used without interruption."

        formatted_messages = [
            {"role": "system", "content": system_prompt},
            *[
                self.message_serializer.serialize_for_llm(msg)
                for msg in filtered_messages
            ],
        ]

        # Generate reasoning response
        response = await self.llm.ainvoke(formatted_messages)
        parsed = self._parse_react_output(response.content)

        # Create reasoning step record
        reasoning_step = {
            "thought": parsed.get("thought", ""),
            "action": parsed.get("action"),
            "action_input": parsed.get("action_input"),
            "timestamp": datetime.now().isoformat(),
        }

        # Prepare state update with preservation
        state_update = {
            "messages": [response],
            "reasoning_steps": state["reasoning_steps"] + [reasoning_step],
            "current_thought": reasoning_step["thought"],
            "iteration_count": state["iteration_count"] + 1,
            "approved_tools": approved_tools,  # Critical preservation
        }

        # Handle final answer
        if reasoning_step["action"] == "Final Answer":
            answer_content = (
                reasoning_step["action_input"].get("answer", "")
                if reasoning_step["action_input"]
                else ""
            )
            state_update["final_answer"] = answer_content

        return self._preserve_critical_state(state, state_update)

    def _preserve_critical_state(
        self, state: AgentState, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ensure critical state fields are preserved across node transitions.

        This method embodies the principle of explicit state management,
        preventing the silent loss of approval memory.
        """
        critical_fields = [
            "approved_tools",
            "rejected_tools",
            "tool_call_mapping",
            "rejection_reasons",
        ]

        for field in critical_fields:
            if field not in updates and field in state:
                updates[field] = state[field]

        return updates

    def _build_guided_system_prompt(self, state: AgentState, guidance: str) -> str:
        """Build system prompt with continuation guidance."""

        base_prompt = ReActPrompts.format_system_prompt(self.tools, self.max_iterations)
        rejected_tools = state.get("rejected_tools", [])

        additional_guidance = f"""

        IMPORTANT CONTEXT:
        - Some tools were rejected: {', '.join(rejected_tools)}
        - Focus on: {guidance}
        - Continue helping the user with what you CAN do
        - Don't mention rejected tools unless necessary
        """

        return base_prompt + additional_guidance

    def _smart_continuation_node(self, state: AgentState) -> Dict[str, Any]:
        """Smart continuation when tools are rejected."""

        pending_tool = state.get("pending_tool_call", {})
        rejected_tool = pending_tool.get("tool_name", "")

        # Get original user request
        original_query = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                original_query = msg.content
                break

        # Track rejected tools
        rejected_tools = state.get("rejected_tools", [])
        if rejected_tool and rejected_tool not in rejected_tools:
            rejected_tools.append(rejected_tool)

        # Ask LLM what else can be done
        continuation_prompt = f"""
        The user asked: "{original_query}"
        The {rejected_tool} tool was rejected.

        What other parts of their request can I still help with?
        Can I provide alternatives for the rejected part?

        Respond with JSON:
        {{
            "can_continue": true/false,
            "user_message": "What to tell the user",
            "next_focus": "What to focus on next"
        }}
        """

        try:
            response = self.llm.invoke(
                [{"role": "user", "content": continuation_prompt}]
            )
            import json

            plan = json.loads(response.content)

            user_message = plan.get("user_message", "Let me help with what I can do.")

            return {
                "messages": [AIMessage(content=user_message)],
                "rejected_tools": rejected_tools,
                "continuation_guidance": plan.get("next_focus"),
                "interrupt_decision": InterruptDecision.NONE.value,
                "pending_tool_call": None,
                "awaiting_tool_approval": False,
            }

        except Exception as e:
            # Fallback
            return {
                "messages": [
                    AIMessage(
                        content=f"I understand the {rejected_tool} tool was rejected. Let me continue with what I can do."
                    )
                ],
                "rejected_tools": rejected_tools,
                "interrupt_decision": InterruptDecision.NONE.value,
                "pending_tool_call": None,
                "awaiting_tool_approval": False,
            }

    def _tool_selector_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Sophisticated tool selection gateway with impeccable interrupt orchestration.

        This node embodies the architectural principle of fail-secure design,
        ensuring every tool invocation respects safety boundaries with elegance.
        """
        # Capture current approval context
        current_interrupt_decision = state.get(
            "interrupt_decision", InterruptDecision.NONE.value
        )
        pending_tool = state.get("pending_tool_call")

        # Elegant handling of resumed approval flows
        if pending_tool and current_interrupt_decision in [
            InterruptDecision.APPROVED.value,
            InterruptDecision.REJECTED.value,
        ]:
            return self._preserve_critical_state(state, {})

        # Extract reasoning intention
        latest_step = state["reasoning_steps"][-1] if state["reasoning_steps"] else None

        if not latest_step:
            logger.warning("No reasoning step found in tool selector")
            return state

        tool_name = latest_step.get("action")
        tool_input = latest_step.get("action_input", {})

        # Validate tool invocation
        if not tool_name or tool_name == "Final Answer":
            return state

        # Rejection history check with semantic precision
        rejected_tools = state.get("rejected_tools", [])
        if tool_name in rejected_tools:
            rejection_reason = state.get("rejection_reasons", {}).get(
                tool_name, "Previously rejected during this session"
            )

            skip_message = (
                f"⚫ Tool '{tool_name}' was previously rejected.\n"
                f"Reason: {rejection_reason}\n"
                f"Continuing with alternative approach."
            )

            logger.info(f"Skipping rejected tool: {tool_name}")

            return self._preserve_critical_state(
                state,
                {
                    "messages": [AIMessage(content=skip_message)],
                },
            )

        # Generate cryptographically unique identifier
        tool_call_id = str(uuid.uuid4())

        # Update invocation mapping
        tool_call_mapping = state.get("tool_call_mapping", {})
        tool_call_mapping[tool_call_id] = tool_name

        # Capture approval context
        approved_tools = state.get("approved_tools", [])

        logger.info(
            f"🔍 Tool Security Evaluation\n"
            f"├─ Tool: {tool_name}\n"
            f"├─ Approved: {approved_tools}\n"
            f"├─ Rejected: {rejected_tools}\n"
            f"└─ Requires Approval: {self.tool_safety_registry.requires_approval(tool_name)}"
        )

        # Pre-approval check
        if tool_name in approved_tools:
            logger.info(f"✅ Tool '{tool_name}' pre-approved - direct execution")
            return self._preserve_critical_state(
                state,
                {
                    "pending_tool_call": {
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "tool_call_id": tool_call_id,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "tool_call_mapping": tool_call_mapping,
                    "interrupt_decision": InterruptDecision.NONE.value,
                    "awaiting_tool_approval": False,
                },
            )

        # CRITICAL: Approval requirement check
        if self.tool_safety_registry.requires_approval(tool_name):
            approval_request = self._craft_approval_request(
                tool_name, tool_input, approved_tools, rejected_tools
            )

            logger.info(f"🔐 INTERRUPT: Tool '{tool_name}' requires approval")

            # Create interrupt state with architectural precision
            interrupt_state = {
                "interrupt_decision": InterruptDecision.PENDING.value,
                "pending_tool_call": {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_call_id": tool_call_id,
                    "timestamp": datetime.now().isoformat(),
                },
                "interrupt_message": approval_request,
                "awaiting_tool_approval": True,
                "tool_call_mapping": tool_call_mapping,
                "messages": [
                    AIMessage(
                        content=approval_request,
                        additional_kwargs={
                            "requires_approval": True,
                            "tool_name": tool_name,
                        },
                    )
                ],
            }

            return self._preserve_critical_state(state, interrupt_state)

        # Non-approval path
        logger.info(f"🟢 Tool '{tool_name}' does not require approval")
        return self._preserve_critical_state(
            state,
            {
                "pending_tool_call": {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_call_id": tool_call_id,
                    "timestamp": datetime.now().isoformat(),
                },
                "tool_call_mapping": tool_call_mapping,
            },
        )

    def _craft_approval_request(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        approved_tools: List[str],
        rejected_tools: List[str],
    ) -> str:
        """
        Craft semantically rich approval requests with contextual awareness.
        """
        base_request = self.tool_safety_registry.generate_approval_request(
            tool_name, tool_input
        )

        context_addendum = "\n\n📊 Current Session Context:\n"

        if approved_tools:
            context_addendum += f"├─ ✅ Approved: {', '.join(approved_tools)}\n"
        else:
            context_addendum += "├─ ✅ Approved: None\n"

        if rejected_tools:
            context_addendum += f"└─ ❌ Rejected: {', '.join(rejected_tools)}"
        else:
            context_addendum += "└─ ❌ Rejected: None"

        return base_request + context_addendum

    async def _approval_checkpoint_node(self, state: AgentState) -> Dict[str, Any]:
        """Approval checkpoint with elegant state transitions and approval memory."""
        interrupt_decision = state.get(
            "interrupt_decision", InterruptDecision.NONE.value
        )

        if interrupt_decision == InterruptDecision.APPROVED.value:
            logger.info("✅ Tool execution approved by human")

            # Extract and remember the approved tool
            pending_tool = state.get("pending_tool_call", {})
            tool_name = pending_tool.get("tool_name")
            approved_tools = state.get("approved_tools", []).copy()  # Create a copy

            if tool_name and tool_name not in approved_tools:
                approved_tools.append(tool_name)
                logger.info(
                    f"📝 Tool '{tool_name}' added to approved list for this session"
                )
                logger.info(f"📋 Current approved tools: {approved_tools}")

            # Clear approval state but maintain workflow continuity
            return self._preserve_critical_state(
                state,
                {
                    "interrupt_decision": InterruptDecision.NONE.value,
                    "awaiting_tool_approval": False,
                    "approved_tools": approved_tools,  # Critical: pass the updated list
                },
            )

        if interrupt_decision == InterruptDecision.REJECTED.value:
            logger.info("❌ Tool execution rejected by human")
            return self._preserve_critical_state(
                state,
                {
                    "interrupt_decision": InterruptDecision.REJECTED.value,
                    "awaiting_tool_approval": False,
                    "approved_tools": state.get(
                        "approved_tools", []
                    ),  # Preserve existing
                },
            )

        return state

    async def _tool_executor_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute approved tools with comprehensive error handling."""
        pending_tool = state.get("pending_tool_call")

        if not pending_tool:
            latest_step = (
                state["reasoning_steps"][-1] if state["reasoning_steps"] else None
            )
            if latest_step:
                tool_call_id = str(uuid.uuid4())
                pending_tool = {
                    "tool_name": latest_step.get("action"),
                    "tool_input": latest_step.get("action_input", {}),
                    "tool_call_id": tool_call_id,
                }

        if not pending_tool or not pending_tool.get("tool_name"):
            return state

        tool_name = pending_tool["tool_name"]
        tool_input = pending_tool["tool_input"]
        tool_call_id = pending_tool.get("tool_call_id", str(uuid.uuid4()))

        # Execute the tool
        tool_executor = self._tool_executor_map.get(tool_name)

        if not tool_executor:
            error_msg = f"Tool '{tool_name}' not found in executor map"
            logger.error(error_msg)
            return self._preserve_critical_state(
                state,
                {
                    "messages": [AIMessage(content=error_msg)],
                    "pending_tool_call": None,
                    "approved_tools": state.get("approved_tools", []),  # Preserve
                },
            )

        try:
            result = await tool_executor.ainvoke(tool_input)

            # Create proper ToolMessage with required tool_call_id
            tool_message = ToolMessage(
                content=str(result), tool_call_id=tool_call_id, name=tool_name
            )

            # CRITICAL: Preserve the approved_tools list
            return self._preserve_critical_state(
                state=state,
                updates={
                    "messages": [tool_message],
                    "pending_tool_call": None,
                    "awaiting_tool_approval": False,
                    "interrupt_decision": InterruptDecision.NONE.value,
                    "approved_tools": state.get(
                        "approved_tools", []
                    ),  # Preserve the list
                },
            )

        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            logger.error(error_msg, exc_info=True)

            return self._preserve_critical_state(
                state=state,
                updates={
                    "messages": [AIMessage(content=error_msg)],
                    "pending_tool_call": None,
                    "awaiting_tool_approval": False,
                    "interrupt_decision": InterruptDecision.NONE.value,
                    "approved_tools": state.get(
                        "approved_tools", []
                    ),  # Preserve even on error
                },
            )

    async def _synthesis_node(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced synthesis with sophisticated output formatting orchestration."""

        # Extract raw final answer with existing elegance
        raw_final_answer = state.get("final_answer", "")

        if not raw_final_answer:
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and "Final Answer" in msg.content:
                    content = msg.contents
                    if "Action Input:" in content:
                        try:
                            json_match = re.search(
                                r"Action Input:\s*({.*?})", content, re.DOTALL
                            )
                            if json_match:
                                action_input = json.loads(json_match.group(1))
                                raw_final_answer = action_input.get("answer", content)
                            else:
                                raw_final_answer = content
                        except:
                            raw_final_answer = content
                    else:
                        raw_final_answer = content
                    break

        if not raw_final_answer:
            raw_final_answer = "I've completed the analysis of your request."

        # Construct tool execution summary with architectural precision
        tool_results = {}
        rejected_tools = state.get("rejected_tools", [])

        for tool_msg in reversed(
            [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
        ):
            if tool_msg.name not in tool_results:
                tool_results[tool_msg.name] = tool_msg.content

        # Build comprehensive response components
        response_components = [raw_final_answer]

        if tool_results:
            # Uncomment to include tool execution results
            response_components.extend(["\n**Tools Executed:**"])
            response_components.extend(
                [
                    f"• ✅ {tool_name}: {result}"
                    for tool_name, result in tool_results.items()
                ]
            )
            # pass

        if rejected_tools:
            response_components.append(
                f"\n**Tools Rejected:** {', '.join(rejected_tools)}"
            )

        enhanced_answer = "\n".join(response_components)

        # CRITICAL: Apply formatting transformation if directive exists
        formatting_prompt = state.get("output_formatting_prompt")

        if formatting_prompt:
            formatted_answer = await self._apply_output_formatting(
                raw_content=enhanced_answer,
                formatting_directive=formatting_prompt,
                execution_context={
                    "tool_results": tool_results,
                    "rejected_tools": rejected_tools,
                    "raw_answer": raw_final_answer,
                },
            )
        else:
            formatted_answer = enhanced_answer

        synthesis_message = AIMessage(
            content=formatted_answer,
            additional_kwargs={
                "synthesized": True,
                "formatted": bool(formatting_prompt),
            },
        )

        return self._preserve_critical_state(
            state,
            {
                "messages": [synthesis_message],
                "raw_final_answer": enhanced_answer,  # Preserve unformatted version
            },
        )

    async def _apply_output_formatting(
        self,
        raw_content: str,
        formatting_directive: str,
        execution_context: Dict[str, Any],
    ) -> str:
        """Apply sophisticated output formatting with contextual awareness."""

        formatting_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert output formatter. Transform the provided content "
                    "according to the user's formatting directive while preserving all "
                    "factual information and semantic meaning. Maintain professional "
                    "accuracy while applying the requested stylistic transformation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"**Original Content:**\n{raw_content}\n\n"
                    f"**Formatting Directive:**\n{formatting_directive}\n\n"
                    f"**Execution Context:**\n"
                    f"- Tools Used: {list(execution_context['tool_results'].keys())}\n"
                    f"- Rejected Tools: {execution_context['rejected_tools']}\n\n"
                    f"Please format the original content according to the directive "
                    f"while preserving all factual accuracy and computational results."
                ),
            },
        ]

        try:
            formatting_response = await self.llm.ainvoke(formatting_messages)
            return formatting_response.content.strip()
        except Exception as e:
            logger.warning(f"Output formatting failed: {e}, returning original content")
            return raw_content

    def _route_from_reasoning(self, state: AgentState) -> str:
        """Route from reasoning with semantic clarity."""
        if state.get("final_answer"):
            return "synthesize"

        latest_step = state["reasoning_steps"][-1] if state["reasoning_steps"] else None

        if latest_step and latest_step.get("action"):
            if latest_step["action"] == "Final Answer":
                return "synthesize"
            else:
                return "select_tool"

        # If no clear action, continue reasoning
        return "continue"

    def _route_from_continuation(self, state: AgentState) -> str:
        """Route from continuation orchestrator."""
        if state.get("continue_to_synthesis"):
            return "complete"
        return "continue_reasoning"

    def _route_from_selector(self, state: AgentState) -> str:
        """
        Sophisticated routing logic with interrupt-aware decision making.

        This method embodies the principle of explicit state transitions,
        ensuring no approval requirement is silently bypassed.
        """
        pending_tool = state.get("pending_tool_call")
        interrupt_decision = state.get(
            "interrupt_decision", InterruptDecision.NONE.value
        )
        awaiting_approval = state.get("awaiting_tool_approval", False)

        # Log routing decision context
        logger.info(
            f"🚦 Routing Decision\n"
            f"├─ Pending Tool: {pending_tool.get('tool_name') if pending_tool else 'None'}\n"
            f"├─ Interrupt Decision: {interrupt_decision}\n"
            f"└─ Awaiting Approval: {awaiting_approval}"
        )

        # CRITICAL: Check if we just set up an interrupt
        if awaiting_approval and interrupt_decision == InterruptDecision.PENDING.value:
            logger.info("🔐 Routing to approval checkpoint for interrupt")
            return "needs_approval"

        # Check pre-approved tools
        if pending_tool:
            tool_name = pending_tool.get("tool_name")
            approved_tools = state.get("approved_tools", [])

            if tool_name in approved_tools:
                logger.info(f"✅ Routing pre-approved tool '{tool_name}' to executor")
                return "execute_directly"

        # Handle post-approval routing
        if (
            pending_tool
            and interrupt_decision == InterruptDecision.APPROVED.value
            and not awaiting_approval
        ):
            return "execute_directly"

        # Check if tool requires approval
        if pending_tool:
            tool_name = pending_tool.get("tool_name")
            if self.tool_safety_registry.requires_approval(tool_name):
                approved_tools = state.get("approved_tools", [])
                if tool_name not in approved_tools:
                    logger.info(f"🔐 Tool '{tool_name}' needs approval")
                    return "needs_approval"

        # Default to execution for tools that don't require approval
        if pending_tool:
            return "execute_directly"

        # Fallback
        logger.warning("No clear routing path determined")
        return "execute_directly"

    def _route_from_approval(self, state: AgentState) -> str:
        """Route from approval checkpoint with clear decision logic."""
        interrupt_decision = state.get(
            "interrupt_decision", InterruptDecision.NONE.value
        )

        if interrupt_decision == InterruptDecision.APPROVED.value:
            return "proceed"
        elif interrupt_decision == InterruptDecision.REJECTED.value:
            return "reject"
        elif state.get("awaiting_tool_approval"):
            return "await_approval"

        return "proceed"

    def _parse_react_output(self, content: str) -> Dict[str, Any]:
        """Parse ReAct-formatted output with robust pattern matching."""
        patterns = {
            "thought": r"Thought:\s*(.*?)(?=\nAction:|$)",
            "action": r"Action:\s*(.*?)(?=\nAction Input:|$)",
            "action_input": r"Action Input:\s*(.*?)(?=\n|$)",
        }

        parsed = {}

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
            if match:
                value = match.group(1).strip()

                if key == "action_input":
                    try:
                        parsed[key] = json.loads(value)
                    except json.JSONDecodeError:
                        json_match = re.search(r"\{.*\}", value, re.DOTALL)
                        if json_match:
                            try:
                                parsed[key] = json.loads(json_match.group(0))
                            except:
                                parsed[key] = {"input": value}
                        else:
                            parsed[key] = {"input": value}
                else:
                    parsed[key] = value

        return parsed

    def create_session_config(
        self,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create properly formatted session configuration."""
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        config = {
            "configurable": {
                "thread_id": session_id,
                "recursion_limit": self.max_iterations or 100,
            }
        }

        if namespace:
            config["configurable"]["checkpoint_ns"] = namespace

        if user_id:
            config["configurable"]["user_id"] = user_id

        return config

    def invoke(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute query with full interrupt support and elegant error handling."""
        if not config:
            config = self.create_session_config(session_id, namespace)

        logger.info(f"Processing query with config: {config}")

        # Check for existing state (resuming from interrupt)
        checkpoint_tuple = self.checkpointer.get_tuple(config)

        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            existing_state = checkpoint_tuple.checkpoint.get("channel_values", {})

            # Check if we're resuming from an interrupt
            if existing_state.get("awaiting_tool_approval"):
                logger.info(
                    "⚠️ Session has pending interrupt - returning interrupt state"
                )
                return {
                    "interrupted": True,
                    "interrupt_message": existing_state.get("interrupt_message"),
                    "pending_tool": existing_state.get("pending_tool_call"),
                    "session_id": config["configurable"]["thread_id"],
                }

            # Check if we're in a continuation state
            if existing_state.get("messages") and not any(
                isinstance(msg, HumanMessage) and msg.content == query
                for msg in existing_state.get("messages", [])
            ):
                # This is a continuation, not a new query
                logger.info("Continuing existing workflow")

                # Simply invoke the workflow without new input
                result = self.workflow.invoke(None, config=config)

                # Handle potential interrupts
                if result.get("awaiting_tool_approval"):
                    return {
                        "interrupted": True,
                        "interrupt_message": result.get("interrupt_message"),
                        "pending_tool": result.get("pending_tool_call"),
                        "session_id": config["configurable"]["thread_id"],
                    }

                return {
                    "messages": result.get("messages", []),
                    "final_answer": result.get("final_answer"),
                    "session_id": config["configurable"]["thread_id"],
                }

        # New query - standard initialization
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "reasoning_steps": [],
            "current_thought": "",
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "final_answer": None,
            "interrupt_decision": InterruptDecision.NONE.value,
            "pending_tool_call": None,
            "interrupt_message": None,
            "awaiting_tool_approval": False,
            "tool_call_mapping": {},
            "rejected_tools": [],
            "rejection_reasons": {},
            "approved_tools": [],
        }

        # Execute workflow
        try:
            result = self.workflow.invoke(initial_state, config=config)

            # Check if we're in an interrupt state
            if result.get("awaiting_tool_approval"):
                logger.info("🛑 Workflow interrupted for approval")
                return {
                    "interrupted": True,
                    "interrupt_message": result.get("interrupt_message"),
                    "pending_tool": result.get("pending_tool_call"),
                    "session_id": config["configurable"]["thread_id"],
                }

            # Normal completion
            return {
                "messages": result.get("messages", []),
                "final_answer": result.get("final_answer"),
                "session_id": config["configurable"]["thread_id"],
            }

        except Exception as e:
            logger.error(f"Workflow execution error: {e}", exc_info=True)
            return {
                "error": str(e),
                "messages": [AIMessage(content=f"An error occurred: {str(e)}")],
                "session_id": config["configurable"]["thread_id"],
            }

    async def ainvoke(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        namespace: Optional[str] = None,
        output_format_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute query with full interrupt support and elegant error handling."""
        if not config:
            config = self.create_session_config(session_id, namespace)

        logger.info(f"Processing query with config: {config}")

        # Check for existing state (resuming from interrupt)
        checkpoint_tuple = await self.checkpointer.aget_tuple(config)

        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            existing_state = checkpoint_tuple.checkpoint.get("channel_values", {})

            # Check if we're resuming from an interrupt
            if existing_state.get("awaiting_tool_approval"):
                logger.info(
                    "⚠️ Session has pending interrupt - returning interrupt state"
                )
                return {
                    "interrupted": True,
                    "interrupt_message": existing_state.get("interrupt_message"),
                    "pending_tool": existing_state.get("pending_tool_call"),
                    "session_id": config["configurable"]["thread_id"],
                }

            # Check if we're in a continuation state
            if existing_state.get("messages") and not any(
                isinstance(msg, HumanMessage) and msg.content == query
                for msg in existing_state.get("messages", [])
            ):
                # This is a continuation, not a new query
                logger.info("Continuing existing workflow")

                # Simply invoke the workflow without new input
                result = await self.workflow.ainvoke(None, config=config)

                # Handle potential interrupts
                if result.get("awaiting_tool_approval"):
                    return {
                        "interrupted": True,
                        "interrupt_message": result.get("interrupt_message"),
                        "pending_tool": result.get("pending_tool_call"),
                        "session_id": config["configurable"]["thread_id"],
                    }

                return {
                    "messages": result.get("messages", []),
                    "final_answer": result.get("final_answer"),
                    "session_id": config["configurable"]["thread_id"],
                }

        # New query - standard initialization
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "reasoning_steps": [],
            "current_thought": "",
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "final_answer": None,
            "interrupt_decision": InterruptDecision.NONE.value,
            "pending_tool_call": None,
            "interrupt_message": None,
            "awaiting_tool_approval": False,
            "tool_call_mapping": {},
            "rejected_tools": [],
            "rejection_reasons": {},
            "approved_tools": [],
            "output_formatting_prompt": output_format_prompt
            or self.default_output_formatter,
            "raw_final_answer": None,
        }

        # Execute workflow
        try:
            result = await self.workflow.ainvoke(initial_state, config=config)

            # Check if we're in an interrupt state
            if result.get("awaiting_tool_approval"):
                logger.info("🛑 Workflow interrupted for approval")
                return {
                    "interrupted": True,
                    "interrupt_message": result.get("interrupt_message"),
                    "pending_tool": result.get("pending_tool_call"),
                    "session_id": config["configurable"]["thread_id"],
                }

            # Normal completion
            return {
                "messages": result.get("messages", []),
                "final_answer": result.get("final_answer"),
                "session_id": config["configurable"]["thread_id"],
            }

        except Exception as e:
            logger.error(f"Workflow execution error: {e}", exc_info=True)
            return {
                "error": str(e),
                "messages": [AIMessage(content=f"An error occurred: {str(e)}")],
                "session_id": config["configurable"]["thread_id"],
            }

    def resume_with_approval(
        self,
        session_id: str,
        approved: bool,
        namespace: Optional[str] = None,
        rejection_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resume interrupted workflow with approval decision and state reconstruction."""
        config = self.create_session_config(session_id, namespace)

        # Retrieve current state for context
        checkpoint_tuple = self.checkpointer.get_tuple(config)
        if not checkpoint_tuple:
            return {"error": "No checkpoint found", "session_id": session_id}

        current_state = checkpoint_tuple.checkpoint.get("channel_values", {})

        # Extract the pending tool to add to approved list if approved
        pending_tool = current_state.get("pending_tool_call", {})
        tool_name = pending_tool.get("tool_name")

        # Get current approved tools list
        approved_tools = current_state.get("approved_tools", []).copy()

        # If approving, add tool to approved list NOW
        if approved and tool_name and tool_name not in approved_tools:
            approved_tools.append(tool_name)
            logger.info(
                f"📝 Pre-adding '{tool_name}' to approved tools list: {approved_tools}"
            )

        # Build state update that preserves context
        state_update = {
            "interrupt_decision": (
                InterruptDecision.APPROVED.value
                if approved
                else InterruptDecision.REJECTED.value
            ),
            # Preserve critical state elements
            "messages": current_state.get("messages", []),
            "reasoning_steps": current_state.get("reasoning_steps", []),
            "tool_call_mapping": current_state.get("tool_call_mapping", {}),
            "rejected_tools": current_state.get("rejected_tools", []),
            "awaiting_tool_approval": True,  # Keep this True to ensure proper routing
            "pending_tool_call": current_state.get("pending_tool_call"),
            "approved_tools": approved_tools,  # Include the updated list
        }

        if not approved and rejection_reason:
            # Add rejection context without disrupting flow
            rejection_msg = SystemMessage(
                content=f"Tool execution rejected: {rejection_reason}. Continuing with alternative approach."
            )
            state_update["messages"] = current_state.get("messages", []) + [
                rejection_msg
            ]

        logger.info(
            f"{'✅ Approving' if approved else '❌ Rejecting'} "
            f"interrupt for session {session_id} with approved_tools: {approved_tools}"
        )

        try:
            # Resume with preserved state
            self.workflow.update_state(config, state_update)
            result = self.workflow.invoke(None, config=config)

            # Check for subsequent interrupts
            if result.get("awaiting_tool_approval"):
                return {
                    "interrupted": True,
                    "interrupt_message": result.get("interrupt_message"),
                    "pending_tool": result.get("pending_tool_call"),
                    "session_id": session_id,
                }

            return {
                "messages": result.get("messages", []),
                "final_answer": result.get("final_answer"),
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(f"Resume error: {e}", exc_info=True)
            return {
                "error": str(e),
                "messages": [AIMessage(content=f"Resume error: {str(e)}")],
                "session_id": session_id,
            }

    async def aresume_with_approval(
        self,
        session_id: str,
        approved: bool,
        namespace: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        edit_input: Optional[Dict[str, Any]] = None,  # New parameter
    ) -> Dict[str, Any]:
        """Resume interrupted workflow with approval decision and optional input editing."""

        config = self.create_session_config(session_id, namespace)
        checkpoint_tuple = await self.checkpointer.aget_tuple(config)

        if not checkpoint_tuple:
            return {"error": "No checkpoint found", "session_id": session_id}

        current_state = checkpoint_tuple.checkpoint.get("channel_values", {})
        pending_tool = current_state.get("pending_tool_call", {})
        tool_name = pending_tool.get("tool_name")
        approved_tools = current_state.get("approved_tools", []).copy()

        # Handle input editing with implicit approval and context injection
        if edit_input is not None:
            if tool_name and tool_name not in approved_tools:
                approved_tools.append(tool_name)
                logger.info(
                    f"📝 Tool '{tool_name}' implicitly approved through editing"
                )

            # Update pending tool call with edited input
            updated_pending_tool = pending_tool.copy()
            updated_pending_tool["tool_input"] = edit_input

            # CRITICAL: Inject edit awareness into message stream
            edit_awareness_message = SystemMessage(
                content=f"EDIT NOTIFICATION: User modified {tool_name} input from "
                f"{json.dumps(pending_tool.get('tool_input', {}))} to {json.dumps(edit_input)}. "
                f"All subsequent reasoning should reference the edited values."
            )

            state_update = {
                "interrupt_decision": InterruptDecision.APPROVED.value,
                "messages": current_state.get("messages", [])
                + [edit_awareness_message],
                "reasoning_steps": current_state.get("reasoning_steps", []),
                "tool_call_mapping": current_state.get("tool_call_mapping", {}),
                "rejected_tools": current_state.get("rejected_tools", []),
                "awaiting_tool_approval": True,
                "pending_tool_call": updated_pending_tool,
                "approved_tools": approved_tools,
                "edited_tool_input": edit_input,
            }

            logger.info(f"🎯 Input edited for '{tool_name}': {edit_input}")
        else:
            # Standard approval/rejection flow
            if approved and tool_name and tool_name not in approved_tools:
                approved_tools.append(tool_name)

            state_update = {
                "interrupt_decision": (
                    InterruptDecision.APPROVED.value
                    if approved
                    else InterruptDecision.REJECTED.value
                ),
                "messages": current_state.get("messages", []),
                "reasoning_steps": current_state.get("reasoning_steps", []),
                "tool_call_mapping": current_state.get("tool_call_mapping", {}),
                "rejected_tools": current_state.get("rejected_tools", []),
                "awaiting_tool_approval": True,
                "pending_tool_call": current_state.get("pending_tool_call"),
                "approved_tools": approved_tools,
            }

            if not approved and rejection_reason:
                rejection_msg = SystemMessage(
                    content=f"Tool execution rejected: {rejection_reason}. Continuing with alternative approach."
                )
                state_update["messages"] = current_state.get("messages", []) + [
                    rejection_msg
                ]

        try:
            await self.workflow.aupdate_state(config, state_update)
            result = await self.workflow.ainvoke(None, config=config)

            if result.get("awaiting_tool_approval"):
                return {
                    "interrupted": True,
                    "interrupt_message": result.get("interrupt_message"),
                    "pending_tool": result.get("pending_tool_call"),
                    "session_id": session_id,
                }

            return {
                "messages": result.get("messages", []),
                "final_answer": result.get("final_answer"),
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(f"Resume error: {e}", exc_info=True)
            return {
                "error": str(e),
                "messages": [AIMessage(content=f"Resume error: {str(e)}")],
                "session_id": session_id,
            }

    async def aget_interrupt_status(
        self, session_id: str, namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check session interrupt status with comprehensive error handling."""
        config = self.create_session_config(session_id, namespace)

        try:
            checkpoint_tuple = await self.checkpointer.aget_tuple(config)

            if not checkpoint_tuple:
                return {"has_interrupt": False, "reason": "No checkpoint found"}

            state = checkpoint_tuple.checkpoint.get("channel_values", {})

            # Only return interrupt if it's still pending
            if (
                state.get("awaiting_tool_approval")
                and state.get("interrupt_decision") != InterruptDecision.APPROVED.value
            ):
                return {
                    "has_interrupt": True,
                    "interrupt_message": state.get("interrupt_message"),
                    "pending_tool": state.get("pending_tool_call"),
                    "session_id": session_id,
                }

            return {"has_interrupt": False}

        except Exception as e:
            logger.error(f"Status check error: {e}")
            return {"has_interrupt": False, "error": str(e)}

    def _build_enhanced_system_prompt(self, state: AgentState) -> str:
        """Build system prompt with rejection context and execution awareness."""
        base_prompt = ReActPrompts.format_system_prompt(self.tools, self.max_iterations)

        # Add rejection context if any tools were rejected
        rejected_tools = state.get("rejected_tools", [])
        if rejected_tools:
            rejection_context = (
                f"\n\nIMPORTANT: The following tools have been rejected by the user "
                f"and should NOT be used: {', '.join(rejected_tools)}. "
                f"Find alternative approaches that don't require these tools."
            )
            base_prompt += rejection_context

        # Add execution context to encourage continuation
        tool_messages = [
            msg for msg in state["messages"] if isinstance(msg, ToolMessage)
        ]
        if tool_messages:
            executed_tools = [msg.name for msg in tool_messages if hasattr(msg, "name")]
            if executed_tools:
                execution_context = (
                    f"\n\nPROGRESS: You have already executed: {', '.join(executed_tools)}. "
                    f"Continue with the remaining tasks from the user's original request."
                )
                base_prompt += execution_context

        return base_prompt

    async def _handle_rejection_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Sophisticated rejection handling with state reset for continued execution.

        This node ensures the workflow maintains momentum after rejection,
        clearing interrupt states to allow subsequent tool evaluations.
        """
        pending_tool = state.get("pending_tool_call", {})
        rejected_tool = pending_tool.get("tool_name")

        # Update rejection tracking
        rejected_tools = state.get("rejected_tools", []).copy()
        rejection_reasons = state.get("rejection_reasons", {}).copy()

        if rejected_tool and rejected_tool not in rejected_tools:
            rejected_tools.append(rejected_tool)
            rejection_reasons[rejected_tool] = "User rejected during approval process"
            logger.info(f"📝 Added '{rejected_tool}' to rejection list")

        # Craft guidance for continued execution
        rejection_message = SystemMessage(
            content=(
                f"Tool '{rejected_tool}' was rejected by the user.\n"
                f"Current rejected tools: {', '.join(rejected_tools)}.\n"
                f"Continue with the remaining tasks using available approved tools."
            )
        )

        # CRITICAL: Clear ALL interrupt-related state
        return self._preserve_critical_state(
            state,
            {
                "messages": [rejection_message],
                "interrupt_decision": InterruptDecision.NONE.value,
                "pending_tool_call": None,
                "awaiting_tool_approval": False,
                "rejected_tools": rejected_tools,
                "rejection_reasons": rejection_reasons,
            },
        )

def _synthesize_video_metadata(
    raw_video_data: str,
    original_query: str,
    result_count: int
) -> str:
    """
    Transform raw YouTube discovery data into semantically enriched intelligence.

    Architected to extract maximal semantic value from YouTube's content
    ecosystem while maintaining elegant presentation aesthetics.
    """
    try:
        # Parse the YouTube URL collection with architectural precision
        video_urls = _extract_video_urls(raw_video_data)

        if not video_urls:
            return f"Content discovery yielded no actionable video intelligence for: {original_query}"

        formatted_intelligence = [
            f"🎥 **YouTube Content Intelligence**: {original_query}",
            f"📊 **Discovery Metrics**: {len(video_urls)} videos synthesized",
            f"🎯 **Content Accessibility**: Direct YouTube integration",
            "",
            "**📋 Discovered Content Inventory:**"
        ]

        for index, video_url in enumerate(video_urls, 1):
            video_id = _extract_video_identifier(video_url)
            enhanced_url = f"https://youtube.com{video_url}"

            formatted_intelligence.append(
                f"{index:2d}. 📺 Video {index} → `{video_id}`\n"
                f"    🔗 {enhanced_url}"
            )

        formatted_intelligence.extend([
            "",
            f"💡 **Content Utilization Strategy**: Each video represents curated content",
            f"🎓 **Educational Value**: Semantically matched to query intent: '{original_query}'"
        ])

        return "\n".join(formatted_intelligence)

    except Exception as synthesis_exception:
        logger.error(f"Video metadata synthesis failure: {synthesis_exception}")
        return f"Raw YouTube Data: {raw_video_data}"

def _extract_video_urls(raw_data: str) -> List[str]:
    """
    Architect elegant URL extraction from YouTube's response ecosystem with precision parsing.
    """
    try:
        # Handle both string representations and actual lists
        if raw_data.startswith('[') and raw_data.endswith(']'):
            # Parse as Python list literal with security consciousness
            video_urls = eval(raw_data)  # Controlled evaluation in tool context
            return [url for url in video_urls if isinstance(url, str) and url.startswith('/watch?v=')]

        # Alternative parsing for comma-separated format
        if ',' in raw_data:
            url_fragments = [fragment.strip().strip("'\"") for fragment in raw_data.split(',')]
            return [url for url in url_fragments if url.startswith('/watch?v=')]

        # Single URL fallback mechanism
        if raw_data.startswith('/watch?v='):
            return [raw_data]

        return []

    except Exception as parsing_exception:
        logger.warning(f"URL extraction complexity: {parsing_exception}")
        # Regex fallback for architectural resilience
        url_pattern = r'/watch\?v=[\w-]+'
        return re.findall(url_pattern, raw_data)
def _extract_video_identifier(youtube_url: str) -> str:
    """
    Synthesize video identification with architectural elegance and semantic precision.
    """
    try:
        # Parse YouTube URL structure with sophisticated pattern recognition
        parsed_url = urlparse(f"https://youtube.com{youtube_url}")
        query_parameters = parse_qs(parsed_url.query)

        video_id = query_parameters.get('v', ['unknown'])[0]
        return video_id[:11] if len(video_id) >= 11 else video_id  # YouTube ID standardization

    except Exception as identification_exception:
        logger.warning(f"Video identification complexity: {identification_exception}")
        # Regex extraction as architectural fallback
        id_match = re.search(r'v=([\w-]+)', youtube_url)
        return id_match.group(1)[:11] if id_match else "unknown"

def _edit_youtube_parameters(parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Orchestrate sophisticated YouTube parameter editing with semantic validation.

    Architected for intelligent content discovery parameter refinement while
    maintaining search intent integrity and result quality optimization.
    """
    edited_configuration = {}

    if "search_query" in parameters:
        current_query = parameters.get("search_query", "")
        print(f"🎥 Current Search Query: {current_query}")

        refined_query = input("🔍 Enter refined search query: ").strip()

        if not refined_query:
            return None

        # Sophisticated query enhancement validation
        if len(refined_query) < 2:
            print("⚠️  Search query requires minimum 2 characters for semantic relevance")
            return None

        edited_configuration["search_query"] = refined_query

    if "result_limit" in parameters:
        current_limit = parameters.get("result_limit", 5)
        print(f"📊 Current Result Limit: {current_limit}")

        new_limit_input = input(f"📈 Enter result limit (1-20) [{current_limit}]: ").strip()

        if new_limit_input.isdigit():
            refined_limit = max(1, min(int(new_limit_input), 20))
            edited_configuration["result_limit"] = refined_limit
        else:
            edited_configuration["result_limit"] = current_limit

    if "academic_domain" in parameters:
        current_domain = parameters.get("academic_domain", "")
        print(f"🎓 Current Academic Domain: {current_domain}")

        refined_domain = input("📚 Enter academic domain: ").strip()

        if refined_domain:
            edited_configuration["academic_domain"] = refined_domain
        else:
            edited_configuration["academic_domain"] = current_domain

    if "content_depth" in parameters:
        current_depth = parameters.get("content_depth", "comprehensive")
        print(f"📊 Current Content Depth: {current_depth}")
        print("Available depths: introductory, comprehensive, advanced")

        refined_depth = input(f"🎯 Select content depth [{current_depth}]: ").strip().lower()

        valid_depths = ["introductory", "comprehensive", "advanced"]
        if refined_depth in valid_depths:
            edited_configuration["content_depth"] = refined_depth
        else:
            edited_configuration["content_depth"] = current_depth

    # Propagate unchanged parameters with architectural precision
    for parameter_key, parameter_value in parameters.items():
        if parameter_key not in edited_configuration:
            edited_configuration[parameter_key] = parameter_value

    return edited_configuration if edited_configuration != parameters else None

@tool
async def orchestrate_youtube_intelligence(
    search_query: str,
    result_limit: int = 5
) -> str:
    """
    Synthesize YouTube content discovery through sophisticated search orchestration.

    Leverages YouTube's content ecosystem to deliver semantically enriched video
    intelligence without API rate limiting constraints, enabling comprehensive
    content exploration through elegant scraping methodologies.

    Args:
        search_query: Semantic search directive for content discovery
        result_limit: Granular control over result density (1-20 videos)

    Returns:
        Architecturally formatted video intelligence with metadata enrichment
    """
    try:
        youtube_orchestrator = YouTubeSearchTool()

        # Validate and sanitize result constraints
        refined_limit = max(1, min(result_limit, 20))
        search_directive = f"{search_query},{refined_limit}"

        video_intelligence = await youtube_orchestrator.ainvoke(search_directive)

        if not video_intelligence or video_intelligence.strip() == "[]":
            return f"No YouTube content intelligence discovered for: {search_query}"

        return _synthesize_video_metadata(video_intelligence, search_query, refined_limit)

    except ImportError:
        return (
            "YouTube intelligence orchestration requires 'youtube-search' package. "
            "Execute: pip install youtube-search"
        )
    except Exception as orchestration_exception:
        logger.error(f"YouTube search orchestration failure: {orchestration_exception}")
        return f"Content discovery error for '{search_query}': {str(orchestration_exception)}"


@tool
async def search_financial_news(ticker_symbol: str) -> str:
    """
    Orchestrate sophisticated financial news retrieval with semantic enrichment.

    Leverages Yahoo Finance's comprehensive financial ecosystem to deliver
    contextually rich news intelligence for equity analysis and market sentiment.

    Args:
        ticker_symbol: Corporate equity identifier (e.g., 'AAPL', 'MSFT', 'NVDA')

    Returns:
        Semantically formatted financial news synthesis with metadata enrichment
    """
    try:
        yahoo_finance_orchestrator = YahooFinanceNewsTool(top_k=5)
        financial_intelligence = await yahoo_finance_orchestrator.ainvoke(ticker_symbol.upper())

        if not financial_intelligence or financial_intelligence.strip() == "":
            return f"No financial news intelligence found for ticker: {ticker_symbol}"

        return f"📈 Financial News Intelligence for {ticker_symbol.upper()}:\n\n{financial_intelligence}"

    except ImportError:
        return (
            "Yahoo Finance integration requires 'yfinance' package. "
            "Execute: pip install yfinance"
        )
    except Exception as financial_exception:
        return f"Financial news retrieval error for {ticker_symbol}: {str(financial_exception)}"


# Enhanced Multi-Symbol Financial Intelligence
@tool
async def analyze_market_sentiment(ticker_symbols: str, max_articles_per_symbol: int = 3) -> str:
    """
    Sophisticated multi-symbol financial sentiment orchestration with comparative analysis.

    Synthesizes market intelligence across multiple equity positions for comprehensive
    portfolio sentiment evaluation and strategic decision intelligence.

    Args:
        ticker_symbols: Comma-separated equity identifiers (e.g., 'AAPL,MSFT,GOOGL')
        max_articles_per_symbol: Granular control over information density per equity

    Returns:
        Comparative financial sentiment synthesis with market positioning intelligence
    """
    try:
        ticker_collection = [symbol.strip().upper() for symbol in ticker_symbols.split(',')]

        if not ticker_collection:
            return "Invalid ticker symbol configuration provided"

        financial_orchestrator = YahooFinanceNewsTool(top_k=max_articles_per_symbol)
        sentiment_synthesis = []

        for equity_symbol in ticker_collection[:5]:  # Architectural constraint for performance
            try:
                market_intelligence = await financial_orchestrator.ainvoke(equity_symbol)

                if market_intelligence and market_intelligence.strip():
                    sentiment_synthesis.append(
                        f"📊 **{equity_symbol}** Market Intelligence:\n"
                        f"{market_intelligence}\n"
                    )
                else:
                    sentiment_synthesis.append(
                        f"📊 **{equity_symbol}**: No current market intelligence available\n"
                    )

            except Exception as symbol_exception:
                sentiment_synthesis.append(
                    f"📊 **{equity_symbol}**: Intelligence retrieval error - {str(symbol_exception)}\n"
                )

        if not sentiment_synthesis:
            return "No financial intelligence synthesized for provided equity symbols"

        return (
            f"🎯 **Multi-Symbol Market Sentiment Analysis**\n"
            f"{'='*60}\n\n" +
            "\n".join(sentiment_synthesis)
        )

    except Exception as orchestration_exception:
        return f"Market sentiment analysis error: {str(orchestration_exception)}"


@tool(infer_schema=True)
async def search_web_duckduckgo(query: str) -> str:
    """Search the web using DuckDuckGo with privacy-focused comprehensive results."""
    try:
        search_engine = DuckDuckGoSearchRun(verbose=True)
        results = await search_engine.ainvoke(query)
        return f"Web Search Results: {results}"
    except Exception as e:
        return f"DuckDuckGo search error: {str(e)}"


@tool
async def search_web_structured(query: str, max_results: int = 5) -> str:
    """Advanced DuckDuckGo search returning structured results with metadata."""
    try:
        search_engine = DuckDuckGoSearchResults(
            max_results=max_results, output_format="json"
        )
        results = await search_engine.ainvoke(query)

        parsed_results = json.loads(results) if isinstance(results, str) else results

        formatted_results = []
        for i, result in enumerate(parsed_results[:max_results], 1):
            formatted_results.append(
                f"{i}. **{result.get('title', 'Unknown')}**\n"
                f"   {result.get('snippet', 'No description')}\n"
                f"   🔗 {result.get('link', 'No link')}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Structured search error: {str(e)}"


@tool
async def search_wikipedia(query: str) -> str:
    """Search Wikipedia for comprehensive information on the given topic."""
    try:
        retriever = WikipediaRetriever(top_k_results=3)
        documents = await retriever.ainvoke(query)

        if not documents:
            return f"No Wikipedia articles found for: {query}"

        results = []
        for doc in documents:
            title = doc.metadata.get("title", "Unknown")
            content = (
                doc.page_content[:500] + "..."
                if len(doc.page_content) > 500
                else doc.page_content
            )
            results.append(f"**{title}:**\n{content}")

        return "\n\n".join(results)

    except Exception as e:
        return f"Wikipedia search error: {str(e)}"


# Tool definitions with proper decorators
@tool
async def search_knowledge_base(query: str) -> str:
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
async def calculate_mathematical_expression(expression: str) -> str:
    """Safely evaluate mathematical expressions."""
    try:
        # Security check
        if any(
            dangerous in expression for dangerous in ["import", "exec", "eval", "__"]
        ):
            return "Error: Expression contains potentially dangerous constructs."

        # Character whitelist
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters."

        # Evaluate safely
        result = eval(expression)
        return f"Mathematical Result: {result}"

    except Exception as e:
        return f"Calculation Error: {str(e)}"


@tool
async def get_current_timestamp() -> str:
    """Retrieve the current timestamp."""
    return f"Current Timestamp: {datetime.now().isoformat()}"


async def demonstrate_sophisticated_interrupt_orchestration(
    query: str = "calculate 5 + 3 * 3, search for Python 3.12 features, and get current time",
) -> None:
    """
    Orchestrate sophisticated human-in-the-loop workflows with cascading interrupt management.

    This method embodies the architectural principle of interrupt transparency,
    enabling seamless multi-tool approval workflows with persistent session context.
    """

    async with AsyncReActAgent(
        llm=AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_AZURE_API_KEY"),
            deployment_name=os.getenv("CHAT_MODELS_DEPLOYMENT_NAME"),
            model_name=os.getenv("CHAT_MODEL_NAME"),
            azure_endpoint=os.getenv("OPENAI_AZURE_API_BASE"),
            api_version=os.getenv("OPENAI_AZURE_API_VERSION"),
            temperature=0.3,
        ),
        max_iterations=12,
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_sophisticated_demo",
        enable_human_interrupts=True,
        use_persistence=True,
    ) as agent:

        session_id = f"multi_interrupt_{uuid.uuid4().hex[:8]}"
        interrupt_sequence = 0

        print(f"🚀 Initiating sophisticated multi-tool workflow")
        print(f"📋 Query: {query}")
        print(f"🆔 Session: {session_id}")

        # Initial invocation with architectural elegance
        current_result = await agent.ainvoke(
            query=query,
            session_id=session_id,
            output_format_prompt="Present results with professional executive formatting",
        )

        # Sophisticated interrupt orchestration loop
        while current_result.get("interrupted"):
            interrupt_sequence += 1

            print(f"\n{'='*60}")
            print(f"⚠️  INTERRUPT #{interrupt_sequence} DETECTED")
            print(f"{'='*60}")

            interrupt_context = current_result.get("interrupt_message", "")
            pending_tool_metadata = current_result.get("pending_tool")

            print(f"\n🔍 Interrupt Context:\n{interrupt_context}")

            if pending_tool_metadata:
                tool_signature = pending_tool_metadata.get("tool_name")
                tool_parameters = pending_tool_metadata.get("tool_input", {})

                print(f"\n🛠️  Tool Awaiting Authorization: {tool_signature}")
                print(f"📊 Current Parameters:")
                print(json.dumps(tool_parameters, indent=2))

            # Sophisticated decision orchestration
            approval_decision = await _orchestrate_interrupt_decision(
                interrupt_sequence, tool_signature, tool_parameters
            )

            match approval_decision["action"]:
                case "approve":
                    print(f"\n✅ Interrupt #{interrupt_sequence}: APPROVED")
                    current_result = await agent.aresume_with_approval(
                        session_id=session_id, approved=True
                    )

                case "edit":
                    print(f"\n✏️  Interrupt #{interrupt_sequence}: EDITING MODE")
                    edited_parameters = approval_decision.get("edited_input")

                    if edited_parameters:
                        print(f"🎯 Applying edited parameters: {edited_parameters}")
                        current_result = await agent.aresume_with_approval(
                            session_id=session_id,
                            approved=True,
                            edit_input=edited_parameters,
                        )
                    else:
                        print("❌ Edit cancelled - no valid parameters provided")
                        break

                case "reject":
                    print(f"\n❌ Interrupt #{interrupt_sequence}: REJECTED")
                    rejection_rationale = approval_decision.get(
                        "reason", "User declined authorization"
                    )
                    current_result = await agent.aresume_with_approval(
                        session_id=session_id,
                        approved=False,
                        rejection_reason=rejection_rationale,
                    )

                case "abort":
                    print(f"\n🛑 Workflow terminated by user directive")
                    return

                case _:
                    print(f"\n⚠️  Invalid decision - defaulting to rejection")
                    current_result = await agent.aresume_with_approval(
                        session_id=session_id,
                        approved=False,
                        rejection_reason="Invalid user input",
                    )

        # Workflow completion orchestration
        print(f"\n{'='*60}")
        print(f"🎯 WORKFLOW COMPLETION")
        print(f"{'='*60}")
        print(f"📊 Total Interrupts Processed: {interrupt_sequence}")

        if current_result.get("messages"):
            print(f"\n📋 Final Synthesized Response:")
            final_messages = current_result["messages"][-3:]

            for message_artifact in final_messages:
                if isinstance(message_artifact, BaseMessage):
                    _display_message_content(message_artifact.content)
                elif isinstance(message_artifact, dict):
                    _display_message_content(
                        message_artifact.get("content", str(message_artifact))
                    )
        else:
            print(
                "⚠️  No final response generated - workflow may have terminated prematurely"
            )


async def _orchestrate_interrupt_decision(
    interrupt_index: int, tool_name: str, tool_parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Sophisticated interrupt decision orchestration with comprehensive parameter editing.

    Returns decision metadata with semantic clarity for workflow continuation.
    """

    print(f"\n🎛️  Decision Matrix for Interrupt #{interrupt_index}")
    print(f"📋 Available Actions: [A]pprove, [E]dit, [R]eject, [Q]uit")

    while True:
        user_directive = input("\n🔘 Select action (A/E/R/Q): ").strip().upper()

        match user_directive:
            case "A" | "APPROVE":
                return {"action": "approve"}

            case "E" | "EDIT":
                edited_parameters = await _orchestrate_parameter_editing(
                    tool_name, tool_parameters
                )
                if edited_parameters:
                    return {"action": "edit", "edited_input": edited_parameters}
                else:
                    print("⚠️  Edit cancelled - returning to decision matrix")
                    continue

            case "R" | "REJECT":
                rejection_reason = input("📝 Rejection reason (optional): ").strip()
                return {
                    "action": "reject",
                    "reason": rejection_reason
                    or f"User rejected {tool_name} execution",
                }

            case "Q" | "QUIT" | "ABORT":
                confirmation = (
                    input("🛑 Confirm workflow termination (y/N): ").strip().upper()
                )
                if confirmation in ["Y", "YES"]:
                    return {"action": "abort"}
                else:
                    continue

            case _:
                print("❌ Invalid input - please select A, E, R, or Q")
                continue


# Update the _orchestrate_parameter_editing function to include financial tools
async def _orchestrate_parameter_editing(
    tool_name: str,
    current_parameters: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Enhanced parameter editing with multimedia intelligence sophistication."""

    print(f"\n✏️  Parameter Editing Interface for: {tool_name}")
    print(f"📊 Current Parameters: {json.dumps(current_parameters, indent=2)}")

    match tool_name:
        case "calculate_mathematical_expression":
            return _edit_mathematical_expression(current_parameters)

        case "search_web_duckduckgo" | "search_web_structured":
            return _edit_search_parameters(current_parameters)

        case "search_financial_news" | "analyze_market_sentiment":
            return _edit_financial_parameters(current_parameters)

        case "orchestrate_youtube_intelligence" | "discover_educational_content":
            return _edit_youtube_parameters(current_parameters)  # Sophisticated multimedia editing

        case tool_name if "search" in tool_name.lower():
            return _edit_search_parameters(current_parameters)

        case _:
            return _edit_generic_parameters(current_parameters)

# Enhanced Parameter Editing for Financial Tools
def _edit_financial_parameters(parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Sophisticated financial parameter editing with semantic validation and intelligence.

    Provides contextual editing interfaces for financial intelligence tools with
    market symbol validation and multi-symbol orchestration capabilities.
    """

    if "ticker_symbol" in parameters:
        current_symbol = parameters.get("ticker_symbol", "")
        print(f"📈 Current Ticker Symbol: {current_symbol}")

        new_symbol = input("💼 Enter new ticker symbol (e.g., AAPL, MSFT): ").strip().upper()

        if not new_symbol:
            return None

        # Sophisticated ticker validation
        if not new_symbol.isalpha() or len(new_symbol) > 5:
            print("⚠️  Invalid ticker symbol format - must be 1-5 alphabetic characters")
            return None

        return {"ticker_symbol": new_symbol}

    elif "ticker_symbols" in parameters:
        current_symbols = parameters.get("ticker_symbols", "")
        print(f"📊 Current Ticker Symbols: {current_symbols}")

        new_symbols = input("💼 Enter ticker symbols (comma-separated, e.g., AAPL,MSFT,GOOGL): ").strip().upper()

        if not new_symbols:
            return None

        # Multi-symbol validation with architectural precision
        symbol_collection = [symbol.strip() for symbol in new_symbols.split(',')]

        if not all(symbol.isalpha() and len(symbol) <= 5 for symbol in symbol_collection):
            print("⚠️  Invalid ticker format - all symbols must be 1-5 alphabetic characters")
            return None

        edited_parameters = {"ticker_symbols": new_symbols}

        # Enhanced configuration for analysis depth
        if "max_articles_per_symbol" in parameters:
            max_articles_input = input(f"📊 Max articles per symbol [{parameters['max_articles_per_symbol']}]: ").strip()
            if max_articles_input.isdigit() and 1 <= int(max_articles_input) <= 10:
                edited_parameters["max_articles_per_symbol"] = int(max_articles_input)
            else:
                edited_parameters["max_articles_per_symbol"] = parameters["max_articles_per_symbol"]

        return edited_parameters

    # Fallback to generic parameter editing for unknown financial tool configurations
    return _edit_generic_parameters(parameters)

def _edit_mathematical_expression(
    parameters: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Elegant mathematical expression editing with validation."""

    current_expression = parameters.get("expression", "")
    print(f"🧮 Current Expression: {current_expression}")

    new_expression = input(
        "🔢 Enter new expression (or press Enter to cancel): "
    ).strip()

    if not new_expression:
        return None

    # Sophisticated expression validation
    try:
        # Basic safety validation
        forbidden_tokens = ["import", "exec", "eval", "__", "getattr", "setattr"]
        if any(token in new_expression for token in forbidden_tokens):
            print("⚠️  Expression contains potentially unsafe constructs")
            return None

        # Syntactic validation without execution
        compile(new_expression, "<string>", "eval")
        return {"expression": new_expression}

    except SyntaxError:
        print("❌ Invalid mathematical expression syntax")
        return None


def _edit_search_parameters(parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Sophisticated search parameter editing with semantic validation."""

    current_query = parameters.get("query", "")
    print(f"🔍 Current Search Query: {current_query}")

    new_query = input("🔎 Enter new search query (or press Enter to cancel): ").strip()

    if not new_query:
        return None

    edited_parameters = {"query": new_query}

    # Handle additional search parameters
    if "max_results" in parameters:
        max_results_input = input(
            f"📊 Max results [{parameters['max_results']}]: "
        ).strip()
        if max_results_input.isdigit():
            edited_parameters["max_results"] = int(max_results_input)
        else:
            edited_parameters["max_results"] = parameters["max_results"]

    return edited_parameters


def _edit_generic_parameters(parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generic parameter editing with dynamic field orchestration."""

    print("🛠️  Generic Parameter Editor")
    edited_parameters = {}

    for parameter_key, current_value in parameters.items():
        print(f"\n📋 Parameter: {parameter_key}")
        print(f"🔘 Current Value: {current_value}")

        new_value = input(f"✏️  New value (press Enter to keep current): ").strip()

        if new_value:
            # Intelligent type inference
            if isinstance(current_value, int) and new_value.isdigit():
                edited_parameters[parameter_key] = int(new_value)
            elif isinstance(current_value, float):
                try:
                    edited_parameters[parameter_key] = float(new_value)
                except ValueError:
                    edited_parameters[parameter_key] = new_value
            else:
                edited_parameters[parameter_key] = new_value
        else:
            edited_parameters[parameter_key] = current_value

    return edited_parameters if edited_parameters != parameters else None


def _display_message_content(content: str) -> None:
    """Elegant message content presentation with semantic formatting."""

    if not content:
        return

    # Sophisticated content formatting based on semantic patterns
    if content.startswith("Thought:"):
        print(f"💭 {content}")
    elif "Tool" in content and "Executed" in content:
        print(f"🔧 {content}")
    elif content.startswith("The result"):
        print(f"📊 {content}")
    else:
        print(f"📝 {content}")


# Enhanced main orchestration
async def main():
    """Sophisticated demonstration orchestration with comprehensive scenarios."""

    scenarios = [
        "Find information about Java launch date and calculate the years since launch",
    ]

    print("🎯 Sophisticated Multi-Interrupt Workflow Demonstration")
    print("=" * 70)

    for scenario_index, scenario_query in enumerate(scenarios, 1):
        print(f"\n🔄 Scenario {scenario_index}:")
        print(f"📋 {scenario_query}")

        try:
            await demonstrate_sophisticated_interrupt_orchestration(scenario_query)
        except KeyboardInterrupt:
            print(f"\n⚠️  Scenario {scenario_index} interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Scenario {scenario_index} failed: {str(e)}")

        print(f"\n{'─' * 50}")


if __name__ == "__main__":
    asyncio.run(main())
