"""Tests for MCP transport protocols (SSE and HTTP)."""

import asyncio
import json
import socket

import httpx
import pytest
from fastapi.testclient import TestClient

from gym_mcp_server.servers import create_app

pytestmark = pytest.mark.timeout(30)

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.sse import sse_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    pytestmark = pytest.mark.skip("MCP library not available")

try:
    import pytest_asyncio

    HAS_PYTEST_ASYNCIO = True
except ImportError:
    HAS_PYTEST_ASYNCIO = False

    # Create a dummy decorator if pytest-asyncio is not available
    class _pytest_asyncio:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

    pytest_asyncio = _pytest_asyncio()


# Expected tool names
EXPECTED_TOOLS = {
    "reset_env",
    "step_env",
    "render_env",
    "get_env_info",
    "close_env",
    "sample_action",
}


def find_free_port():
    """Find a free port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture
def app():
    """Create a FastAPI app for testing."""
    return create_app(env_id="CartPole-v1")


@pytest.fixture
def test_client(app):
    """Create a test client."""
    return TestClient(app)


@pytest_asyncio.fixture
async def test_server(app):
    """Create a test server running in the background."""
    if not HAS_PYTEST_ASYNCIO:
        pytest.skip("pytest-asyncio not available")
    import uvicorn
    from threading import Thread

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    def run_server():
        asyncio.run(server.serve())

    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    max_wait = 10
    for _ in range(max_wait):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/health", timeout=1.0)
                if response.status_code == 200:
                    break
        except Exception:
            await asyncio.sleep(0.5)
    else:
        pytest.fail("Server failed to start")

    try:
        yield base_url
    finally:
        # Cleanup is handled by daemon thread
        pass


class TestMCPHTTPTransport:
    """Test MCP HTTP (streamable-http) transport."""

    @pytest.mark.asyncio
    async def test_http_endpoint_exists(self, test_client):
        """Test that HTTP MCP endpoint exists."""
        # MCP endpoints typically return 406 for GET without proper headers
        # or 400 for missing session ID, but any response means endpoint exists
        response = test_client.get("/mcp")
        assert response.status_code in [200, 400, 406, 405]

    @pytest.mark.asyncio
    async def test_http_list_tools(self, test_server):
        """Test listing tools via HTTP transport."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP library not available")

        base_url = test_server

        # Test HTTP transport
        async with streamablehttp_client(f"{base_url}/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List tools
                tools_result = await session.list_tools()
                tool_names = {tool.name for tool in tools_result.tools}

                # Verify expected tools are present
                assert EXPECTED_TOOLS.issubset(
                    tool_names
                ), f"Missing tools: {EXPECTED_TOOLS - tool_names}"

                # Verify tool definitions
                for tool in tools_result.tools:
                    assert tool.name, "Tool must have a name"
                    assert tool.description, "Tool must have a description"
                    assert tool.inputSchema, "Tool must have input schema"


class TestMCPSSETransport:
    """Test MCP SSE (Server-Sent Events) transport."""

    @pytest.mark.asyncio
    async def test_sse_endpoint_exists(self, test_client):
        """Test that SSE MCP endpoint exists.

        Note: SSE endpoints hang waiting for connections, so we use a short timeout.
        The timeout itself confirms the endpoint exists (404 would return immediately).
        """
        import threading
        import queue

        # SSE endpoints hang waiting for SSE connections when accessed via GET
        # Use a thread with timeout to test this
        result_queue = queue.Queue()

        def make_request():
            try:
                response = test_client.get("/sse")
                result_queue.put(("response", response.status_code))
            except Exception as e:
                result_queue.put(("error", str(e)))

        thread = threading.Thread(target=make_request, daemon=True)
        thread.start()
        thread.join(timeout=2.0)  # 2 second timeout

        if thread.is_alive():
            # Thread is still running (timeout) - this confirms endpoint exists
            # (404 would return immediately)
            return

        # Got a result
        result_type, result_value = result_queue.get_nowait()
        if result_type == "response":
            # If we get a response, check it's not 404
            assert result_value != 404, "SSE endpoint should exist"
            # Common status codes for SSE endpoints without proper setup
            assert result_value in [
                200,
                400,
                406,
                405,
                500,
            ], f"Unexpected status code: {result_value}"
        # Errors other than 404 are also acceptable

    @pytest.mark.asyncio
    async def test_sse_list_tools(self, test_server):
        """Test listing tools via SSE transport."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP library not available")

        base_url = test_server

        # Test SSE transport
        async with sse_client(f"{base_url}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List tools
                tools_result = await session.list_tools()
                tool_names = {tool.name for tool in tools_result.tools}

                # Verify expected tools are present
                assert EXPECTED_TOOLS.issubset(
                    tool_names
                ), f"Missing tools: {EXPECTED_TOOLS - tool_names}"

                # Verify tool definitions
                for tool in tools_result.tools:
                    assert tool.name, "Tool must have a name"
                    assert tool.description, "Tool must have a description"
                    assert tool.inputSchema, "Tool must have input schema"

    @pytest.mark.asyncio
    async def test_sse_call_tool(self, test_server):
        """Test calling a tool via SSE transport."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP library not available")

        base_url = test_server

        async with sse_client(f"{base_url}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Test reset_env tool
                result = await session.call_tool("reset_env", arguments={})
                assert result.content, "Tool call must return content"
                assert (
                    len(result.content) > 0
                ), "Tool call must have at least one content item"

                # Parse result
                result_text = result.content[0].text
                assert result_text, f"Result text is empty: {result_text}"
                result_data = json.loads(result_text)

                # Verify response structure
                assert result_data.get("success") is True, "Tool call should succeed"
                assert "observation" in result_data, "Reset should return observation"
                assert "info" in result_data, "Reset should return info"


class TestMCPTransportComparison:
    """Test that both transports work equivalently."""

    @pytest.mark.asyncio
    async def test_both_transports_list_same_tools(self, test_server):
        """Test that both transports return the same tools."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP library not available")

        base_url = test_server

        # Get tools via HTTP
        http_tools = set()
        async with streamablehttp_client(f"{base_url}/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                http_tools = {tool.name for tool in tools_result.tools}

        # Get tools via SSE
        sse_tools = set()
        async with sse_client(f"{base_url}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                sse_tools = {tool.name for tool in tools_result.tools}

        # Both should return the same tools
        assert (
            http_tools == sse_tools
        ), f"HTTP tools: {http_tools}, SSE tools: {sse_tools}"
        assert EXPECTED_TOOLS.issubset(http_tools), "HTTP missing expected tools"
        assert EXPECTED_TOOLS.issubset(sse_tools), "SSE missing expected tools"

    @pytest.mark.asyncio
    async def test_both_transports_call_tools_equivalently(self, test_server):
        """Test that both transports produce equivalent tool call results."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP library not available")

        base_url = test_server

        # Call get_env_info via HTTP
        http_result = None
        async with streamablehttp_client(f"{base_url}/mcp") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("get_env_info", arguments={})
                result_text = result.content[0].text
                assert result_text, f"HTTP result text is empty: {result_text}"
                http_result = json.loads(result_text)

        # Call get_env_info via SSE
        sse_result = None
        async with sse_client(f"{base_url}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("get_env_info", arguments={})
                result_text = result.content[0].text
                assert result_text, f"SSE result text is empty: {result_text}"
                sse_result = json.loads(result_text)

        # Both should return equivalent results
        assert http_result.get("success") == sse_result.get("success")
        assert http_result.get("env_info") == sse_result.get("env_info")
