"""A tiny real MCP server (official SDK, FastMCP) used as a stdio upstream in the
integration test. Run: python mini_mcp_server.py  (speaks MCP over stdio)."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mini")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the given text back to the caller."""
    return text


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


if __name__ == "__main__":
    mcp.run()   # stdio transport by default
