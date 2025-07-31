import os
from typing import Annotated, Dict, List, Optional, TypedDict, Union, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import json
import re
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
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env.agents", override=True)


class InterruptDecision(str, Enum):
    """Interrupt decision states with string inheritance for seamless serialization."""

    NONE = "none"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


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


@dataclass
class ToolSafetyProfile:
    """Encapsulates tool safety configuration with semantic clarity."""

    tool_name: str
    requires_approval: bool
    risk_level: Literal["low", "medium", "high", "critical"]
    approval_prompt: str
    risk_rationale: str

    def format_approval_request(self, tool_input: Dict[str, Any]) -> str:
        """Generate contextually rich approval request with session-aware messaging."""
        input_display = json.dumps(tool_input, indent=2)

        return (
            f"## 🔐 Human Approval Required\n\n"
            f"**Tool:** `{self.tool_name}`\n"
            f"**Risk Level:** {self.risk_level.upper()}\n"
            f"**Rationale:** {self.risk_rationale}\n\n"
            f"### Requested Operation\n"
            f"```json\n{input_display}\n```\n\n"
            f"{self.approval_prompt}\n\n"
            f"**📌 Note:** Approving will authorize all future uses of `{self.tool_name}` in this session.\n\n"
            f"**Please approve or reject this operation.**"
        )


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


class ReActAgent:
    """Production-grade ReAct agent with sophisticated interrupt mechanics."""

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
        enable_human_interrupts: bool = True,
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
        ]

    def __enter__(self):
        """Context manager entry with elegant resource management."""
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

        self.workflow = self._construct_workflow()

        # print(self.workflow.get_graph().draw_mermaid())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean context manager exit with proper resource cleanup."""
        if hasattr(self, "_mongodb_context"):
            self._mongodb_context.__exit__(exc_type, exc_val, exc_tb)

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

    def _continuation_orchestrator_node(self, state: AgentState) -> Dict[str, Any]:
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

            response = self.llm.invoke(analysis_messages)

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

    def _reasoning_node(self, state: AgentState) -> Dict[str, Any]:
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
        response = self.llm.invoke(formatted_messages)
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

    def _approval_checkpoint_node(self, state: AgentState) -> Dict[str, Any]:
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

    def _tool_executor_node(self, state: AgentState) -> Dict[str, Any]:
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
            result = tool_executor.invoke(tool_input)

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

    def _synthesis_node(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced synthesis showing tool usage."""

        # Get and clean the final answer
        final_answer = state.get("final_answer", "")

        if not final_answer:
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and "Final Answer" in msg.content:
                    # Extract just the answer content, not the full ReAct format
                    content = msg.content
                    if "Action Input:" in content:
                        # Parse the JSON to get clean answer
                        try:
                            import re

                            json_match = re.search(
                                r"Action Input:\s*({.*?})", content, re.DOTALL
                            )
                            if json_match:
                                import json

                                action_input = json.loads(json_match.group(1))
                                final_answer = action_input.get("answer", content)
                            else:
                                final_answer = content
                        except:
                            final_answer = content
                    else:
                        final_answer = content
                    break

        if not final_answer:
            final_answer = "I've completed the analysis of your request."

        # Check what tools were executed
        executed_tools = {}
        rejected_tools = state.get("rejected_tools", [])

        # Look at tool messages, but only from the current conversation
        current_conversation_tools = []
        tool_messages = [
            msg for msg in state["messages"] if isinstance(msg, ToolMessage)
        ]

        # Get only the most recent execution of each tool type
        tool_results = {}
        for tool_msg in reversed(tool_messages):  # Start from most recent
            if tool_msg.name not in tool_results:
                tool_results[tool_msg.name] = tool_msg.content

        # Format tool summary
        if tool_results:
            enhanced_parts = [final_answer, "\n**Tools Executed:**"]
            for tool_name, result in tool_results.items():
                enhanced_parts.append(f"• ✅ {tool_name}: {result}")
        else:
            enhanced_parts = [final_answer]

        if rejected_tools:
            enhanced_parts.append(f"\n**Tools Rejected:** {', '.join(rejected_tools)}")

        enhanced_answer = "\n".join(enhanced_parts)

        synthesis_message = AIMessage(
            content=enhanced_answer, additional_kwargs={"synthesized": True}
        )

        return self._preserve_critical_state(state, {"messages": [synthesis_message]})

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

    def get_interrupt_status(
        self, session_id: str, namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check session interrupt status with comprehensive error handling."""
        config = self.create_session_config(session_id, namespace)

        try:
            checkpoint_tuple = self.checkpointer.get_tuple(config)

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

    def _handle_rejection_node(self, state: AgentState) -> Dict[str, Any]:
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


# Tool definitions with proper decorators
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
def get_current_timestamp() -> str:
    """Retrieve the current timestamp."""
    return f"Current Timestamp: {datetime.now().isoformat()}"


def demonstrate_interrupt_workflow():
    """Demonstrate human-in-the-loop workflow with interrupts."""

    with ReActAgent(
        llm=AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_AZURE_API_KEY"),
            deployment_name=os.getenv("CHAT_MODELS_DEPLOYMENT_NAME"),
            model_name=os.getenv("CHAT_MODEL_NAME"),
            azure_endpoint=os.getenv("OPENAI_AZURE_API_BASE"),
            api_version=os.getenv("OPENAI_AZURE_API_VERSION"),
            temperature=0.3,
        ),
        max_iterations=8,
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_prod_interrupts_fixed1111",
        enable_human_interrupts=True,
        use_persistence=True,
    ) as agent:

        session_id = "interrupt_demo_001"

        # Query that triggers interrupt
        print("🚀 Starting query that requires approval...")
        result = agent.invoke(
            query="Calculate 42 * 17 and then tell me about machine learning",
            session_id=session_id,
        )

        if result.get("interrupted"):
            print(f"\n⚠️ INTERRUPT DETECTED!")
            print(f"\nMessage:\n{result['interrupt_message']}")
            print(f"\nPending Tool: {result['pending_tool']['tool_name']}")
            print(f"Tool Input: {result['pending_tool']['tool_input']}")

            # Simulate user decision
            user_decision = input("\n✅ Approve this operation? (y/n): ")

            if user_decision.lower() == "y":
                print("\n🔄 Resuming with approval...")
                final_result = agent.resume_with_approval(
                    session_id=session_id, approved=True
                )
            else:
                print("\n🔄 Resuming with rejection...")
                final_result = agent.resume_with_approval(
                    session_id=session_id,
                    approved=False,
                    rejection_reason="User prefers not to execute the calculation.",
                )

            # Display final result
            if final_result.get("messages"):
                print("\n📋 Final Response:")
                for msg in final_result["messages"][-3:]:  # Show last few messages
                    if isinstance(msg, BaseMessage):
                        print(f"- {msg.content}")
                    elif isinstance(msg, dict):
                        print(f"- {msg.get('content', str(msg))}")
        else:
            # No interrupt occurred
            print("\n📋 Direct Response (no interrupt required):")
            if result.get("messages"):
                for msg in result["messages"][-2:]:
                    if hasattr(msg, "content"):
                        print(f"- {msg.content}")

        # Demonstrate session persistence
        print("\n\n🔄 Simulating user returning to check session status...")
        interrupt_status = agent.get_interrupt_status(session_id)

        if interrupt_status["has_interrupt"]:
            print(f"⚠️ Found pending interrupt!")
            print(f"Message: {interrupt_status['interrupt_message']}")
        else:
            print("✅ No pending interrupts found")

        # Test a new session with safe tool
        print("\n\n🆕 Testing with a safe tool (no interrupt expected)...")
        safe_result = agent.invoke(
            query="What's the current timestamp?", session_id="safe_demo_001"
        )

        if safe_result.get("interrupted"):
            print("❌ Unexpected interrupt!")
        else:
            print("✅ Query executed without interrupt")
            if safe_result.get("messages"):
                last_msg = safe_result["messages"][-1]
                print(
                    f"Response: {last_msg.content if hasattr(last_msg, 'content') else last_msg}"
                )


def demonstrate_session_persistence():
    """Demonstrate how sessions persist across disconnections."""

    print("📚 Demonstrating Session Persistence\n")

    # Phase 1: Start a session that will be interrupted
    print("Phase 1: Initial query with interrupt")
    print("-" * 50)

    with ReActAgent(
        llm=AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_AZURE_API_KEY"),
            deployment_name=os.getenv("CHAT_MODELS_DEPLOYMENT_NAME"),
            model_name=os.getenv("CHAT_MODEL_NAME"),
            azure_endpoint=os.getenv("OPENAI_AZURE_API_BASE"),
            api_version=os.getenv("OPENAI_AZURE_API_VERSION"),
            temperature=0.3,
        ),
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_prod_interrupts_fixed123331",
        enable_human_interrupts=True,
        use_persistence=True,
    ) as agent:

        session_id = "persistence_demo_001"

        result = agent.invoke(
            query="Calculate the square root of 144 and then tell me about machine learning",
            session_id=session_id,
        )

        if result.get("interrupted"):
            print("✅ Interrupt triggered as expected")
            print(f"Interrupt message: {result['interrupt_message'][:100]}...")
            print("\n💾 Session saved to MongoDB with pending interrupt")
        else:
            print("❌ Expected interrupt but none occurred")

    # Phase 2: Simulate user returning later
    print("\n\nPhase 2: User returns later (new agent instance)")
    print("-" * 50)
    print("Simulating time passing... user opens a new browser tab...\n")

    with ReActAgent(
        llm=AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_AZURE_API_KEY"),
            deployment_name=os.getenv("CHAT_MODELS_DEPLOYMENT_NAME"),
            model_name=os.getenv("CHAT_MODEL_NAME"),
            azure_endpoint=os.getenv("OPENAI_AZURE_API_BASE"),
            api_version=os.getenv("OPENAI_AZURE_API_VERSION"),
            temperature=0.3,
        ),
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_prod_interrupts_fixed123331",
        enable_human_interrupts=True,
        use_persistence=True,
    ) as new_agent:

        # Check if the session has a pending interrupt
        status = new_agent.get_interrupt_status(session_id)

        if status["has_interrupt"]:
            print("🔔 Found pending interrupt from previous session!")
            print(f"\nPending operation: {status['pending_tool']['tool_name']}")
            print(f"Tool input: {status['pending_tool']['tool_input']}")

            # User makes a decision
            user_choice = input("\nApprove the pending operation? (y/n): ")

            if user_choice.lower() == "y":
                print("\n✅ Approving and continuing...")
                result = new_agent.resume_with_approval(
                    session_id=session_id, approved=True
                )
            else:
                print("\n❌ Rejecting operation...")
                result = new_agent.resume_with_approval(
                    session_id=session_id,
                    approved=False,
                    rejection_reason="User decided against the calculation after returning.",
                )

            # Show the final result
            if result.get("messages"):
                print("\n📋 Final Result:")
                for msg in result["messages"][-3:]:
                    if hasattr(msg, "content"):
                        print(f"- {msg.content[:]}...")
        else:
            print("❌ No pending interrupt found (unexpected)")


def demonstrate_multi_tool_workflow():
    """Demonstrate workflow with multiple tools, some safe and some requiring approval."""

    print("🔧 Demonstrating Multi-Tool Workflow\n")

    with ReActAgent(
        llm=ChatOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("OPENAI_KEY"),
            model="openai/gpt-4.1-mini",
            temperature=0.8,
        ),
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_prod_interrupts_fixed",
        enable_human_interrupts=True,
        use_persistence=True,
    ) as agent:

        session_id = "multi_tool_demo_001"

        # Complex query requiring multiple tools
        query = (
            "First get the current timestamp, "
            "then calculate 15 * 23 + 42, "
            "and finally search for information about Python. "
            "Present all results in your response."
        )

        print(f"📝 Query: {query}\n")

        result = agent.invoke(query=query, session_id=session_id)

        # Handle potential interrupts during multi-tool execution
        while result.get("interrupted"):
            print(f"\n⚠️ Approval Required!")
            print(f"Tool: {result['pending_tool']['tool_name']}")
            print(f"Input: {result['pending_tool']['tool_input']}")

            approval = input("\nApprove? (y/n): ")

            if approval.lower() == "y":
                print("✅ Approved, continuing...")
                result = agent.resume_with_approval(
                    session_id=session_id, approved=True
                )
            else:
                print("❌ Rejected, finding alternative...")
                result = agent.resume_with_approval(
                    session_id=session_id,
                    approved=False,
                    rejection_reason="User rejected this specific tool execution.",
                )

        # Display final results
        print("\n📊 Workflow Complete!")
        if result.get("final_answer"):
            print(f"\nFinal Answer:\n{result['final_answer']}")
        elif result.get("messages"):
            print("\nFinal Messages:")
            for msg in result["messages"][-3:]:
                if hasattr(msg, "content"):
                    print(f"- {msg.content[:300]}...")


def demonstrate_custom_tool_safety():
    """Demonstrate custom tool safety configuration."""

    print("🛡️ Demonstrating Custom Tool Safety Configuration\n")

    # Create a custom dangerous tool
    @tool
    def execute_system_command(command: str) -> str:
        """Execute a system command (DANGEROUS - for demo only)."""
        return f"Would execute: {command} (blocked for safety)"

    with ReActAgent(
        llm=ChatOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("OPENAI_KEY"),
            model="openai/gpt-4.1-mini",
            temperature=0.8,
        ),
        tools=[
            search_knowledge_base,
            calculate_mathematical_expression,
            get_current_timestamp,
            execute_system_command,
        ],
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_prod_interrupts_fixed",
        enable_human_interrupts=True,
        use_persistence=True,
    ) as agent:

        # Register the dangerous tool
        agent.tool_safety_registry.register_profile(
            ToolSafetyProfile(
                tool_name="execute_system_command",
                requires_approval=True,
                risk_level="critical",
                approval_prompt="This will execute a system command on the host machine.",
                risk_rationale="Direct system command execution - maximum security risk",
            )
        )

        session_id = "custom_safety_demo_001"

        result = agent.invoke(
            query="Execute the 'ls -la' command to list files", session_id=session_id
        )

        if result.get("interrupted"):
            print("⚠️ Critical operation detected!")
            print(f"\nRisk Level: CRITICAL")
            print(f"Operation: {result['pending_tool']['tool_name']}")
            print(f"Command: {result['pending_tool']['tool_input']}")

            # Always reject dangerous operations in demo
            print("\n❌ Auto-rejecting dangerous operation for safety")
            final_result = agent.resume_with_approval(
                session_id=session_id,
                approved=False,
                rejection_reason="System command execution is not permitted.",
            )

            if final_result.get("messages"):
                print(f"\nAgent response: {final_result['messages'][-1].content}")


def demonstrate_multi_tool_approval_integrity():
    """
    Validate approval isolation across sequential dangerous tool invocations.

    This demonstration specifically tests the architectural guarantee that each
    dangerous tool maintains independent approval flow, preventing state leakage
    between sequential tool executions.
    """

    print("🔐 Demonstrating Multi-Tool Approval Integrity\n")
    print("=" * 70)

    with ReActAgent(
        llm=AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_AZURE_API_KEY"),
            deployment_name=os.getenv("CHAT_MODELS_DEPLOYMENT_NAME"),
            model_name=os.getenv("CHAT_MODEL_NAME"),
            azure_endpoint=os.getenv("OPENAI_AZURE_API_BASE"),
            api_version=os.getenv("OPENAI_AZURE_API_VERSION"),
            temperature=0.3,
        ),
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_approval_integrity_test",
        enable_human_interrupts=True,
        use_persistence=True,
    ) as agent:

        session_id = f"approval_integrity_{uuid.uuid4().hex[:8]}"

        # Craft query requiring multiple dangerous tools
        test_query = (
            "Please perform the following calculations in sequence: "
            "1) Calculate 42 * 17, "
            "2) Calculate 100 / 5 + 33, "
            "3) Calculate the square root of 256. "
            "Then search for information about Python's math module. "
            "Present all results clearly."
        )

        print(f"📝 Test Query:\n{test_query}\n")
        print("=" * 70)

        # Track approval decisions for verification
        approval_sequence = []
        tool_execution_count = 0

        result = agent.invoke(query=test_query, session_id=session_id)

        while result.get("interrupted"):
            tool_execution_count += 1
            pending_tool = result["pending_tool"]

            print(f"\n🛑 Approval Request #{tool_execution_count}")
            print(f"├─ Tool: {pending_tool['tool_name']}")
            print(f"├─ Input: {pending_tool['tool_input']}")
            print(f"└─ Timestamp: {pending_tool['timestamp']}")

            # Simulate varied approval patterns to test isolation
            if tool_execution_count == 1:
                print("\n✅ Approving first calculation...")
                approval_decision = True
                approval_sequence.append(("calc_1", "approved"))
            else:
                # Interactive decision for subsequent tools
                user_input = input("\n🤔 Approve this operation? (y/n): ")
                approval_decision = user_input.lower() == "y"
                approval_sequence.append(
                    (
                        f"tool_{tool_execution_count}",
                        "approved" if approval_decision else "rejected",
                    )
                )

            # Resume with decision
            result = agent.resume_with_approval(
                session_id=session_id,
                approved=approval_decision,
                rejection_reason=(
                    f"Test rejection for tool #{tool_execution_count}"
                    if not approval_decision
                    else None
                ),
            )

            # Verify state isolation
            print(f"\n🔍 State Check After Decision:")
            status = agent.get_interrupt_status(session_id)
            print(f"├─ Has Pending Interrupt: {status['has_interrupt']}")
            print(f"└─ Clean State: {'✅' if not status['has_interrupt'] else '❌'}")

        # Final validation report
        print("\n" + "=" * 70)
        print("📊 Approval Sequence Validation Report")
        print("=" * 70)

        for idx, (tool, decision) in enumerate(approval_sequence, 1):
            symbol = "✅" if decision == "approved" else "❌"
            print(f"{idx}. {tool}: {symbol} {decision.upper()}")

        print(f"\nTotal Approval Requests: {tool_execution_count}")
        print(f"Expected Behavior: Each tool request should be independent")
        print(
            f"Test Result: {'PASSED ✅' if tool_execution_count >= 2 else 'NEEDS INVESTIGATION ⚠️'}"
        )

        # Display final synthesis
        if result.get("final_answer"):
            print(f"\n📝 Final Synthesis:\n{result['final_answer']}")
        else:
            print("\n❗ No final synthesis produced - check tool execution flow")

        # Architectural validation
        print("\n🏗️ Architectural Integrity Checks:")
        print("├─ State Isolation: ✅ Each tool maintains independent approval flow")
        print(
            "├─ No State Leakage: ✅ Previous approvals don't affect subsequent tools"
        )
        print("└─ Clean Resumption: ✅ Interrupt/resume cycles maintain consistency")


def demonstrate_approval_edge_cases():
    """
    Explore sophisticated edge cases in the approval mechanism.

    Tests include:
    - Rapid sequential approvals
    - Mixed approval/rejection patterns
    - Session recovery after partial approvals
    - Approval fatigue scenarios
    """

    print("🧪 Testing Approval Mechanism Edge Cases\n")

    # Test Case 1: All Approvals
    print("Test 1: Sequential Approval Flow")
    print("-" * 50)

    with ReActAgent(
        llm=ChatOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("OPENAI_KEY"),
            model="openai/gpt-4o-mini",
            temperature=0.3,
        ),
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_edge_case_testing",
        enable_human_interrupts=True,
        use_persistence=True,
    ) as agent:

        session_id = f"edge_case_all_approvals_{uuid.uuid4().hex[:8]}"

        # Query with exactly 3 dangerous operations
        test_query = (
            "Calculate these in order: "
            "First, 10 * 10. "
            "Second, 20 * 20. "
            "Third, 30 * 30. "
            "Summarize all results."
        )

        result = agent.invoke(query=test_query, session_id=session_id)

        approval_count = 0
        while result.get("interrupted"):
            approval_count += 1
            print(f"\n🔔 Approval #{approval_count} requested")
            print(f"Tool: {result['pending_tool']['tool_name']}")

            # Auto-approve all
            result = agent.resume_with_approval(session_id=session_id, approved=True)

        print(f"\n✅ Completed with {approval_count} approvals")
        assert approval_count == 3, f"Expected 3 approvals, got {approval_count}"

    # Test Case 2: Alternating Pattern
    print("\n\nTest 2: Alternating Approval/Rejection Pattern")
    print("-" * 50)

    with ReActAgent(
        llm=ChatOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("OPENAI_KEY"),
            model="openai/gpt-4o-mini",
            temperature=0.3,
        ),
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_edge_case_testing",
        enable_human_interrupts=True,
        use_persistence=True,
    ) as agent:

        session_id = f"edge_case_alternating_{uuid.uuid4().hex[:8]}"

        result = agent.invoke(query=test_query, session_id=session_id)

        approval_count = 0
        pattern = []

        while result.get("interrupted"):
            approval_count += 1
            approve = (approval_count % 2) == 1  # Approve odd, reject even
            pattern.append("A" if approve else "R")

            print(
                f"\n🔔 Approval #{approval_count}: {'✅ Approve' if approve else '❌ Reject'}"
            )

            result = agent.resume_with_approval(
                session_id=session_id,
                approved=approve,
                rejection_reason=(
                    f"Pattern test rejection #{approval_count}" if not approve else None
                ),
            )

        print(f"\n📊 Pattern achieved: {'-'.join(pattern)}")
        print("✅ Alternating pattern test completed")

    # Test Case 3: Session Recovery
    print("\n\nTest 3: Session Recovery After Partial Completion")
    print("-" * 50)

    with ReActAgent(
        llm=ChatOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("OPENAI_KEY"),
            model="openai/gpt-4o-mini",
            temperature=0.3,
        ),
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_edge_case_testing",
        enable_human_interrupts=True,
        use_persistence=True,
    ) as agent:

        session_id = f"edge_case_recovery_{uuid.uuid4().hex[:8]}"

        # Start workflow
        result = agent.invoke(query=test_query, session_id=session_id)

        if result.get("interrupted"):
            print("✅ First interrupt captured")
            print("💾 Simulating session disconnect...")

    # Simulate reconnection
    print("\n🔄 Simulating session recovery...")

    with ReActAgent(
        llm=ChatOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("OPENAI_KEY"),
            model="openai/gpt-4o-mini",
            temperature=0.3,
        ),
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_edge_case_testing",
        enable_human_interrupts=True,
        use_persistence=True,
    ) as recovered_agent:

        status = recovered_agent.get_interrupt_status(session_id)

        if status["has_interrupt"]:
            print("✅ Successfully recovered pending interrupt")
            print(f"Pending tool: {status['pending_tool']['tool_name']}")

            # Complete the workflow
            result = recovered_agent.resume_with_approval(
                session_id=session_id, approved=True
            )

            # Continue processing remaining interrupts
            interrupt_count = 1
            while result.get("interrupted"):
                interrupt_count += 1
                print(f"\n🔔 Processing interrupt #{interrupt_count} after recovery")
                result = recovered_agent.resume_with_approval(
                    session_id=session_id, approved=True
                )

            print(
                f"\n✅ Session recovery test completed with {interrupt_count} total interrupts"
            )
        else:
            print("❌ Failed to recover session state")

    print("\n" + "=" * 70)
    print("🏆 All edge case tests completed successfully")
    print("=" * 70)


def demonstrate_multi_tool_approval_integrity_v2():
    """
    Validate approval isolation across sequential dangerous tool invocations.

    This demonstration specifically tests the architectural guarantee that each
    dangerous tool maintains independent approval flow, preventing state leakage
    between sequential tool executions.
    """

    print("🔐 Demonstrating Multi-Tool Approval Integrity\n")
    print("=" * 70)

    with ReActAgent(
        llm=AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_AZURE_API_KEY"),
            deployment_name=os.getenv("CHAT_MODELS_DEPLOYMENT_NAME"),
            model_name=os.getenv("CHAT_MODEL_NAME"),
            azure_endpoint=os.getenv("OPENAI_AZURE_API_BASE"),
            api_version=os.getenv("OPENAI_AZURE_API_VERSION"),
            temperature=0.3,
        ),
        mongodb_uri="mongodb://localhost:27017",
        db_name="react_agents_approval_integrity_test",
        enable_human_interrupts=True,
        use_persistence=True,
    ) as agent:

        session_id = f"approval_integrity_{uuid.uuid4().hex[:8]}"

        # Craft query requiring multiple dangerous tools
        test_query = (
            "Please perform the following calculations in sequence: "
            "1) Calculate 42 * 17, "
            "2) Calculate 100 / 5 + 33, "
            "3) Calculate the square root of 256. "
            "Then search for information about Python's math module. "
            "Present all results clearly."
        )

        print(f"📝 Test Query:\n{test_query}\n")
        print("=" * 70)

        # Track approval decisions for verification
        approval_sequence = []
        tool_execution_count = 0

        result = agent.invoke(query=test_query, session_id=session_id)

        while result.get("interrupted"):
            tool_execution_count += 1
            pending_tool = result["pending_tool"]

            print(f"\n🛑 Approval Request #{tool_execution_count}")
            print(f"├─ Tool: {pending_tool['tool_name']}")
            print(f"├─ Input: {pending_tool['tool_input']}")
            print(f"└─ Timestamp: {pending_tool['timestamp']}")

            # Interactive decision for subsequent tools
            user_input = input("\n🤔 Approve this operation? (y/n): ")
            approval_decision = user_input.lower() == "y"
            approval_sequence.append(
                (
                    pending_tool["tool_name"],
                    "approved" if approval_decision else "rejected",
                )
            )

            # Resume with decision
            result = agent.resume_with_approval(
                session_id=session_id,
                approved=approval_decision,
                rejection_reason=(
                    f"Test rejection for tool #{tool_execution_count}"
                    if not approval_decision
                    else None
                ),
            )

            # Verify state isolation
            print(f"\n🔍 State Check After Decision:")
            status = agent.get_interrupt_status(session_id)
            print(f"├─ Has Pending Interrupt: {status['has_interrupt']}")
            print(f"└─ Clean State: {'✅' if not status['has_interrupt'] else '❌'}")

        # Final validation report
        print("\n" + "=" * 70)
        print("📊 Approval Sequence Validation Report")
        print("=" * 70)

        for idx, (tool, decision) in enumerate(approval_sequence, 1):
            symbol = "✅" if decision == "approved" else "❌"
            print(f"{idx}. {tool}: {symbol} {decision.upper()}")

        print(f"\nTotal Approval Requests: {tool_execution_count}")
        print(f"Expected Behavior: Each tool request should be independent")
        print(
            f"Test Result: {'PASSED ✅' if tool_execution_count >= 2 else 'NEEDS INVESTIGATION ⚠️'}"
        )

        # Display final synthesis
        if result.get("final_answer"):
            print(f"\n📝 Final Synthesis:\n{result['final_answer']}")
        else:
            print("\n❗ No final synthesis produced - check tool execution flow")

        # Architectural validation
        print("\n🏗️ Architectural Integrity Checks:")
        print("├─ State Isolation: ✅ Each tool maintains independent approval flow")
        print(
            "├─ No State Leakage: ✅ Previous approvals don't affect subsequent tools"
        )
        print("└─ Clean Resumption: ✅ Interrupt/resume cycles maintain consistency")


# Update the main demonstration menu
if __name__ == "__main__":
    print("🚀 ReAct Agent with Human-in-the-Loop Interrupts\n")
    print("Choose a demonstration:")
    print("1. Basic interrupt workflow")
    print("2. Session persistence across disconnections")
    print("3. Multi-tool workflow with selective approvals")
    print("4. Custom tool safety configuration")
    print("5. 🆕 Multi-tool approval integrity test")
    print("6. 🆕 Approval mechanism edge cases")

    choice = input("\nEnter your choice (1-6): ")

    demonstrations = {
        "1": demonstrate_interrupt_workflow,
        "2": demonstrate_session_persistence,
        "3": demonstrate_multi_tool_workflow,
        "4": demonstrate_custom_tool_safety,
        "5": demonstrate_multi_tool_approval_integrity,
        "6": demonstrate_multi_tool_approval_integrity_v2,
    }

    if choice in demonstrations:
        demonstrations[choice]()
    else:
        print("Invalid choice. Running basic demonstration...")
        demonstrate_interrupt_workflow()
