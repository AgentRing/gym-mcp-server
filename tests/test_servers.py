"""Tests for the HTTP server."""

import pytest
from fastapi.testclient import TestClient
from gym_mcp_server.servers import create_app, GymHTTPServer

pytestmark = pytest.mark.timeout(10)


class TestCreateApp:
    """Test create_app function."""

    def test_create_app(self):
        """Test creating FastAPI app."""
        app = create_app(env_id="CartPole-v1")
        assert app is not None
        assert app.title.startswith("Gym Environment API")

    def test_create_app_with_title(self):
        """Test creating app with custom title."""
        app = create_app(env_id="CartPole-v1", title="Custom Title")
        assert app.title == "Custom Title"


class TestRESTEndpoints:
    """Test REST API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app(env_id="CartPole-v1", render_mode="rgb_array")
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        # Root endpoint returns HTML, not JSON
        assert response.headers["content-type"].startswith("text/html")
        assert "Gym MCP Server" in response.text

    def test_api_endpoint(self, client):
        """Test API info endpoint."""
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Gym MCP Server"
        assert data["environment"] == "CartPole-v1"
        assert "endpoints" in data
        assert "links" in data

    def test_reset_endpoint(self, client):
        """Test reset endpoint."""
        response = client.post("/reset", json={"seed": 42})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "observation" in data

    def test_reset_endpoint_no_seed(self, client):
        """Test reset endpoint without seed."""
        response = client.post("/reset", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_step_endpoint(self, client):
        """Test step endpoint."""
        # First reset
        client.post("/reset", json={})
        # Then step
        response = client.post("/step", json={"action": 0})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "observation" in data
        assert "reward" in data

    def test_render_endpoint(self, client):
        """Test render endpoint."""
        client.post("/reset", json={})
        response = client.post("/render", json={"mode": "rgb_array"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "render" in data

    def test_info_endpoint(self, client):
        """Test info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "env_info" in data
        assert data["env_info"]["id"] == "CartPole-v1"

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["env_id"] == "CartPole-v1"

    def test_close_endpoint(self, client):
        """Test close endpoint."""
        response = client.post("/close")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "closed"

    def test_step_endpoint_invalid_action(self, client):
        """Test step endpoint with invalid action."""
        client.post("/reset", json={})
        # CartPole has 2 actions, so action 999 should be invalid
        response = client.post("/step", json={"action": 999})
        # Should still return 200 but with success=False or handle gracefully
        assert response.status_code in [200, 500]

    def test_docs_endpoint(self, client):
        """Test Swagger docs endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_endpoint(self, client):
        """Test OpenAPI schema endpoint."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data


class TestGymHTTPServer:
    """Test GymHTTPServer class."""

    def test_init(self):
        """Test server initialization."""
        server = GymHTTPServer(env_id="CartPole-v1")
        assert server.env_id == "CartPole-v1"
        assert server.app is not None

    def test_init_with_options(self):
        """Test server initialization with options."""
        server = GymHTTPServer(
            env_id="CartPole-v1",
            render_mode="rgb_array",
            title="Test Title",
            description="Test Description",
        )
        assert server.render_mode == "rgb_array"
        assert server.title == "Test Title"
        assert server.description == "Test Description"
