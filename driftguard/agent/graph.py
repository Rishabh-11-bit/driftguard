"""
LangGraph agent graph for DriftGuard.

The graph implements a stateful, memory-augmented conversational agent with
DriftGuard's write pipeline intercepting every memory write.

Graph nodes (in execution order):
    1. retrieve_memory   — semantic search of long-term memory
    2. llm_call          — generate a response given context + memories
    3. extract_writes    — identify new facts/preferences from the response
    4. write_pipeline    — classify + enforce each candidate write
    5. respond           — return the final answer to the user

Author: Aryaan Sethi (1032231670) — Objective 1 & 4
"""

from __future__ import annotations

import logging
import re
from typing import Any, Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from driftguard.agent.memory_backend import MemoryBackend, get_memory_backend
from driftguard.agent.llm_interface import get_llm
from driftguard.provenance.write_pipeline import WritePipeline, CandidateWrite
from driftguard.provenance.enforcement import EnforcementPolicy, HumanReviewQueue

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Agent state
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    session_id: str
    messages: list[BaseMessage]
    retrieved_memories: list[dict[str, Any]]
    candidate_writes: list[str]
    write_results: list[dict[str, Any]]
    final_response: str


# ─────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful, personalised AI assistant with persistent memory.
You recall and use past information about the user to give relevant, personalised responses.

When you learn something new about the user (a preference, fact, or correction), output it
on a separate line at the end of your response in this exact format:
  MEMORY_WRITE: <the fact or preference to remember>

You may output multiple MEMORY_WRITE lines if you learned multiple things.
Do NOT output MEMORY_WRITE for things you already know or for transient context.

Relevant memories from past sessions:
{memories}
"""

MEMORY_WRITE_PATTERN = re.compile(r"^MEMORY_WRITE:\s*(.+)$", re.MULTILINE)


# ─────────────────────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────────────────────

def _make_retrieve_node(memory_backend: MemoryBackend, k: int = 5):
    def retrieve_memory(state: AgentState) -> dict:
        last_user_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break
        memories = memory_backend.search(last_user_msg, k=k) if last_user_msg else []
        logger.debug(f"[Agent] Retrieved {len(memories)} memories.")
        return {"retrieved_memories": memories}
    return retrieve_memory


def _make_llm_node(llm: Any):
    def llm_call(state: AgentState) -> dict:
        memories = state.get("retrieved_memories", [])
        memory_text = "\n".join(
            f"- {m['text']}" for m in memories if not m.get("metadata", {}).get("flagged")
        ) or "No relevant memories yet."

        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(memories=memory_text))
        messages = [system_msg] + state["messages"]

        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)
        logger.debug(f"[Agent] LLM response: {response_text[:80]}…")
        return {"final_response": response_text}
    return llm_call


def _make_extract_node():
    def extract_writes(state: AgentState) -> dict:
        response = state.get("final_response", "")
        writes = MEMORY_WRITE_PATTERN.findall(response)
        logger.debug(f"[Agent] Extracted {len(writes)} candidate writes.")
        return {"candidate_writes": writes}
    return extract_writes


def _make_pipeline_node(pipeline: WritePipeline):
    def write_pipeline_node(state: AgentState) -> dict:
        results = []
        last_user_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break

        for text in state.get("candidate_writes", []):
            candidate = CandidateWrite(
                content=text,
                origin="agent_reflection",
                session_id=state["session_id"],
                triggering_text=last_user_msg,
            )
            result = pipeline.process(candidate)
            results.append(result.to_dict())
        return {"write_results": results}
    return write_pipeline_node


def _make_respond_node():
    def respond(state: AgentState) -> dict:
        # Strip MEMORY_WRITE lines from the user-visible response
        raw = state.get("final_response", "")
        clean = MEMORY_WRITE_PATTERN.sub("", raw).strip()
        return {"final_response": clean}
    return respond


# ─────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────

def build_agent_graph(
    memory_backend: MemoryBackend | None = None,
    llm: Any | None = None,
    pipeline: WritePipeline | None = None,
    k_retrieve: int = 5,
):
    """
    Build and compile the DriftGuard LangGraph agent.

    Parameters
    ----------
    memory_backend : MemoryBackend | None
        Vector store backend. Defaults to FAISSMemoryBackend.
    llm : Any | None
        LangChain chat model. Defaults to get_llm().
    pipeline : WritePipeline | None
        DriftGuard write pipeline. Defaults to WritePipeline(memory_backend).
    k_retrieve : int
        Number of memories to retrieve per query.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph StateGraph ready to invoke.
    """
    memory_backend = memory_backend or get_memory_backend()
    llm = llm or get_llm()
    pipeline = pipeline or WritePipeline(memory_backend=memory_backend)

    graph = StateGraph(AgentState)

    graph.add_node("retrieve_memory",  _make_retrieve_node(memory_backend, k_retrieve))
    graph.add_node("llm_call",         _make_llm_node(llm))
    graph.add_node("extract_writes",   _make_extract_node())
    graph.add_node("write_pipeline",   _make_pipeline_node(pipeline))
    graph.add_node("respond",          _make_respond_node())

    graph.set_entry_point("retrieve_memory")
    graph.add_edge("retrieve_memory", "llm_call")
    graph.add_edge("llm_call",        "extract_writes")
    graph.add_edge("extract_writes",  "write_pipeline")
    graph.add_edge("write_pipeline",  "respond")
    graph.add_edge("respond",         END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────────────────

def run_agent(
    user_message: str,
    session_id: str,
    compiled_graph,
    message_history: list[BaseMessage] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Run a single turn of the DriftGuard agent.

    Returns
    -------
    (response_text, write_results)
    """
    history = list(message_history or [])
    history.append(HumanMessage(content=user_message))

    state = AgentState(
        session_id=session_id,
        messages=history,
        retrieved_memories=[],
        candidate_writes=[],
        write_results=[],
        final_response="",
    )

    final_state = compiled_graph.invoke(state)
    return final_state["final_response"], final_state.get("write_results", [])
