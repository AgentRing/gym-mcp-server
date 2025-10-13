"""
Tests for gym_mcp_server.schemas module.
"""

from gym_mcp_server.schemas import (
    RESET_RESPONSE_SCHEMA,
    STEP_RESPONSE_SCHEMA,
    RENDER_RESPONSE_SCHEMA,
    ENV_INFO_RESPONSE_SCHEMA,
    CLOSE_RESPONSE_SCHEMA,
    TOOL_SCHEMAS,
)


class TestResponseSchemas:
    """Test cases for response schemas."""

    def test_reset_response_schema_structure(self):
        """Test RESET_RESPONSE_SCHEMA structure."""
        assert RESET_RESPONSE_SCHEMA["type"] == "object"
        assert "properties" in RESET_RESPONSE_SCHEMA
        assert "required" in RESET_RESPONSE_SCHEMA

        properties = RESET_RESPONSE_SCHEMA["properties"]
        assert "observation" in properties
        assert "info" in properties
        assert "done" in properties

        required = RESET_RESPONSE_SCHEMA["required"]
        assert "observation" in required
        assert "info" in required
        assert "done" in required

    def test_reset_response_schema_properties(self):
        """Test RESET_RESPONSE_SCHEMA property details."""
        properties = RESET_RESPONSE_SCHEMA["properties"]

        # Test observation property
        assert "description" in properties["observation"]

        # Test info property
        assert properties["info"]["type"] == "object"
        assert "description" in properties["info"]

        # Test done property
        assert properties["done"]["type"] == "boolean"
        assert "description" in properties["done"]

    def test_step_response_schema_structure(self):
        """Test STEP_RESPONSE_SCHEMA structure."""
        assert STEP_RESPONSE_SCHEMA["type"] == "object"
        assert "properties" in STEP_RESPONSE_SCHEMA
        assert "required" in STEP_RESPONSE_SCHEMA

        properties = STEP_RESPONSE_SCHEMA["properties"]
        assert "observation" in properties
        assert "reward" in properties
        assert "done" in properties
        assert "info" in properties

        required = STEP_RESPONSE_SCHEMA["required"]
        assert "observation" in required
        assert "reward" in required
        assert "done" in required
        assert "info" in required

    def test_step_response_schema_properties(self):
        """Test STEP_RESPONSE_SCHEMA property details."""
        properties = STEP_RESPONSE_SCHEMA["properties"]

        # Test observation property
        assert "description" in properties["observation"]

        # Test reward property
        assert properties["reward"]["type"] == "number"
        assert "description" in properties["reward"]

        # Test done property
        assert properties["done"]["type"] == "boolean"
        assert "description" in properties["done"]

        # Test info property
        assert properties["info"]["type"] == "object"
        assert "description" in properties["info"]

    def test_render_response_schema_structure(self):
        """Test RENDER_RESPONSE_SCHEMA structure."""
        assert RENDER_RESPONSE_SCHEMA["type"] == "object"
        assert "properties" in RENDER_RESPONSE_SCHEMA
        assert "required" in RENDER_RESPONSE_SCHEMA

        properties = RENDER_RESPONSE_SCHEMA["properties"]
        assert "render" in properties
        assert "mode" in properties
        assert "type" in properties
        assert "shape" in properties

        required = RENDER_RESPONSE_SCHEMA["required"]
        assert "render" in required
        assert "mode" in required

    def test_render_response_schema_properties(self):
        """Test RENDER_RESPONSE_SCHEMA property details."""
        properties = RENDER_RESPONSE_SCHEMA["properties"]

        # Test render property
        assert "description" in properties["render"]

        # Test mode property
        assert properties["mode"]["type"] == "string"
        assert "description" in properties["mode"]

        # Test type property
        assert properties["type"]["type"] == "string"
        assert "description" in properties["type"]

        # Test shape property
        assert properties["shape"]["type"] == "array"
        assert properties["shape"]["items"]["type"] == "integer"
        assert "description" in properties["shape"]

    def test_env_info_response_schema_structure(self):
        """Test ENV_INFO_RESPONSE_SCHEMA structure."""
        assert ENV_INFO_RESPONSE_SCHEMA["type"] == "object"
        assert "properties" in ENV_INFO_RESPONSE_SCHEMA

        properties = ENV_INFO_RESPONSE_SCHEMA["properties"]
        assert "id" in properties
        assert "action_space" in properties
        assert "observation_space" in properties
        assert "reward_range" in properties
        assert "action_space_size" in properties
        assert "action_space_shape" in properties
        assert "observation_space_shape" in properties
        assert "observation_space_spaces" in properties

    def test_env_info_response_schema_properties(self):
        """Test ENV_INFO_RESPONSE_SCHEMA property details."""
        properties = ENV_INFO_RESPONSE_SCHEMA["properties"]

        # Test id property
        assert properties["id"]["type"] == "string"
        assert "description" in properties["id"]

        # Test action_space property
        assert properties["action_space"]["type"] == "string"
        assert "description" in properties["action_space"]

        # Test observation_space property
        assert properties["observation_space"]["type"] == "string"
        assert "description" in properties["observation_space"]

        # Test reward_range property
        assert properties["reward_range"]["type"] == "array"
        assert properties["reward_range"]["items"]["type"] == "number"
        assert "description" in properties["reward_range"]

        # Test action_space_size property
        assert properties["action_space_size"]["type"] == "integer"
        assert "description" in properties["action_space_size"]

        # Test action_space_shape property
        assert properties["action_space_shape"]["type"] == "array"
        assert properties["action_space_shape"]["items"]["type"] == "integer"
        assert "description" in properties["action_space_shape"]

        # Test observation_space_shape property
        assert properties["observation_space_shape"]["type"] == "array"
        assert properties["observation_space_shape"]["items"]["type"] == "integer"
        assert "description" in properties["observation_space_shape"]

        # Test observation_space_spaces property
        assert properties["observation_space_spaces"]["type"] == "object"
        assert "description" in properties["observation_space_spaces"]

    def test_close_response_schema_structure(self):
        """Test CLOSE_RESPONSE_SCHEMA structure."""
        assert CLOSE_RESPONSE_SCHEMA["type"] == "object"
        assert "properties" in CLOSE_RESPONSE_SCHEMA
        assert "required" in CLOSE_RESPONSE_SCHEMA

        properties = CLOSE_RESPONSE_SCHEMA["properties"]
        assert "status" in properties

        required = CLOSE_RESPONSE_SCHEMA["required"]
        assert "status" in required

    def test_close_response_schema_properties(self):
        """Test CLOSE_RESPONSE_SCHEMA property details."""
        properties = CLOSE_RESPONSE_SCHEMA["properties"]

        # Test status property
        assert properties["status"]["type"] == "string"
        assert "description" in properties["status"]


class TestToolSchemas:
    """Test cases for tool schemas."""

    def test_tool_schemas_structure(self):
        """Test TOOL_SCHEMAS structure."""
        assert isinstance(TOOL_SCHEMAS, dict)
        assert "reset_env" in TOOL_SCHEMAS
        assert "step_env" in TOOL_SCHEMAS
        assert "render_env" in TOOL_SCHEMAS
        assert "close_env" in TOOL_SCHEMAS
        assert "get_env_info" in TOOL_SCHEMAS

    def test_reset_env_tool_schema(self):
        """Test reset_env tool schema."""
        schema = TOOL_SCHEMAS["reset_env"]

        assert schema["name"] == "reset_env"
        assert "description" in schema
        assert "inputSchema" in schema

        input_schema = schema["inputSchema"]
        assert input_schema["type"] == "object"
        assert "properties" in input_schema

        properties = input_schema["properties"]
        assert "seed" in properties
        assert properties["seed"]["type"] == "integer"
        assert "description" in properties["seed"]

    def test_step_env_tool_schema(self):
        """Test step_env tool schema."""
        schema = TOOL_SCHEMAS["step_env"]

        assert schema["name"] == "step_env"
        assert "description" in schema
        assert "inputSchema" in schema

        input_schema = schema["inputSchema"]
        assert input_schema["type"] == "object"
        assert "properties" in input_schema
        assert "required" in input_schema

        properties = input_schema["properties"]
        assert "action" in properties
        assert "description" in properties["action"]

        required = input_schema["required"]
        assert "action" in required

    def test_render_env_tool_schema(self):
        """Test render_env tool schema."""
        schema = TOOL_SCHEMAS["render_env"]

        assert schema["name"] == "render_env"
        assert "description" in schema
        assert "inputSchema" in schema

        input_schema = schema["inputSchema"]
        assert input_schema["type"] == "object"
        assert "properties" in input_schema

        properties = input_schema["properties"]
        assert "mode" in properties
        assert properties["mode"]["type"] == "string"
        assert "description" in properties["mode"]
        # Note: 'default' is not required in the schema as mode can be None

    def test_close_env_tool_schema(self):
        """Test close_env tool schema."""
        schema = TOOL_SCHEMAS["close_env"]

        assert schema["name"] == "close_env"
        assert "description" in schema
        assert "inputSchema" in schema

        input_schema = schema["inputSchema"]
        assert input_schema["type"] == "object"
        assert "properties" in input_schema

        properties = input_schema["properties"]
        assert properties == {}

    def test_get_env_info_tool_schema(self):
        """Test get_env_info tool schema."""
        schema = TOOL_SCHEMAS["get_env_info"]

        assert schema["name"] == "get_env_info"
        assert "description" in schema
        assert "inputSchema" in schema

        input_schema = schema["inputSchema"]
        assert input_schema["type"] == "object"
        assert "properties" in input_schema

        properties = input_schema["properties"]
        assert properties == {}
