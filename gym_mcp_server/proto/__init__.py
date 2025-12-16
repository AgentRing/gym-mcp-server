"""
Proto module for gRPC service definitions.

The Python code is generated from gym_service.proto using:
    python -m grpc_tools.protoc -I gym_mcp_server/proto \
        --python_out=gym_mcp_server --grpc_python_out=gym_mcp_server \
        gym_mcp_server/proto/gym_service.proto

Pydantic models are also generated from the proto definitions:
    python scripts/generate_pydantic_models.py

This generates models.py in this directory for use with FastAPI/HTTP servers.
"""

