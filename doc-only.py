pip install langchain-mcp-adpaters langchain-core
pip install "deepmcpagent[deep]"

from langchain_mcp import MCPAdapter
from langchain_core.agents import AgentExecutor
# For stdio transport
mcp_adapter = MCPAdapter(
server_command=["python", "mcp_server.py"], # Replace with your
server's command
transport_type="stdio"
)
# For Streamable HTTP transport (if your DeepAgent platform supports
it)
# mcp_adapter = MCPAdapter(
# server_url="https://api.example.com/mcp", # Replace with your
server's URL
# transport_type="http"
# )

await mcp_adapter.connect()
tools = await mcp_adapter.get_tools()

from langchain.agents import initialize_agent
from langchain_openai import OpenAI # Example LLM
llm = OpenAI() # Initialize your chosen LLM
agent = initialize_agent(tools, llm,
agent="zero-shot-react-description", verbose=True)
# For DeepAgents, the integration would be similar, ensuring the
tools are accessible
# within the DeepAgent's configuration or execution context.
agent.run("Use the MCP tool to perform a specific task.")
