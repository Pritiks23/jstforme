from __future__ import annotations
from typing import Optional
import asyncio
import logging

from deepagents import create_deep_agent
from langchain_core.runnables import Runnable
from langchain_mcp import MCPAdapter, MCPAdapterError
from .llm import get_mc1_model
from .memory import CairoMemoryTools
from .tools.search import internet_search
from .tools.recommendation import (
    set_weights_tool,
    boost_creator_tool,
    demote_creator_tool,
    block_tag_tool,
    unblock_tag_tool,
    search_content_tool,
    trending_content_tool,
    personalized_feed_tool,
)
from .policy import guard_tools

logger = logging.getLogger(__name__)

CAIRO_SYSTEM_INSTRUCTIONS = """
You are CAIRO - ColomboAI In-App Reactive Operator - an in-app agent that is context-aware, privacy-respectful, and action-oriented.
You operate within the ColomboAI ecosystem (GenAI, Feed, CAIRO, News, Generative Shop).

Mandatory restrictions (NON-NEGOTIABLE):
- You MUST NOT like posts, auto-like, comment on posts, or auto-comment. Never take such actions or suggest them.
- When acting on social surfaces, only read, summarize, plan, draft, schedule (with user confirmation), or recommend.
- Do not publish externally without explicit confirmation.

Recommendation Engine Control:
- You CAN control the recommendation engine via tools to set weights, boost/demote creators, and block/unblock tags.
- You CAN help users discover content using search, trending content, and personalized feeds.
- Prefer small, reversible changes and explain the expected impact when you act.
- Log what you changed and why (in your reply) so humans can review.

General guidance:
- Use long-term memory (Mem0) when helpful.
- Prefer concise, structured answers and include sources for web findings.
- For complex tasks, first write a short plan, then execute step-by-step.
"""

async def build_cairo_agent(builtin_tools: Optional[list[str]] = None) -> Runnable:
    # ------------------------
    # CAIRO memory + local tools
    # ------------------------
    mem_tools = CairoMemoryTools()
    local_tools = [
        internet_search,
        mem_tools.add_tool,
        mem_tools.search_tool,
        mem_tools.get_all_tool,
        set_weights_tool,
        boost_creator_tool,
        demote_creator_tool,
        block_tag_tool,
        unblock_tag_tool,
        search_content_tool,
        trending_content_tool,
        personalized_feed_tool,
    ]
    local_tools = guard_tools(local_tools)

    # ------------------------
    # Optional MCP Adapter integration
    # ------------------------
    mcp_tools = []
    try:
        mcp_adapter = MCPAdapter(
            server_command=["python", "mcp_server.py"],  # adjust if needed
            transport_type="stdio"
        )
        await mcp_adapter.connect()
        mcp_tools = await mcp_adapter.get_tools()
        logger.info(f"MCP tools loaded: {[t.name for t in mcp_tools]}")
    except (MCPAdapterError, FileNotFoundError, ConnectionError) as e:
        logger.warning(f"MCP server unavailable, proceeding with local tools only: {e}")

    # ------------------------
    # Combine local + MCP tools
    # ------------------------
    all_tools = local_tools + mcp_tools

    # ------------------------
    # Initialize DeepAgent
    # ------------------------
    model = get_mc1_model(temperature=0.2, max_tokens=2048)
    agent = create_deep_agent(
        tools=all_tools,
        instructions=CAIRO_SYSTEM_INSTRUCTIONS,
        model=model,
        builtin_tools=builtin_tools,
    )
    return agent

# Helper to run outside asyncio context
def build_cairo_agent_sync(builtin_tools: Optional[list[str]] = None) -> Runnable:
    return asyncio.run(build_cairo_agent(builtin_tools=builtin_tools))
