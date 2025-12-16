#!/usr/bin/env python3
"""
Script to generate Python code from proto files and Pydantic models.

Usage:
    python scripts/generate_proto.py
"""

import subprocess
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent
proto_dir = project_root / "gym_mcp_server" / "proto"
proto_file = proto_dir / "gym_service.proto"

if not proto_file.exists():
    print(f"Error: Proto file not found at {proto_file}")
    sys.exit(1)

# Generate Python code from proto
cmd = [
    sys.executable,
    "-m",
    "grpc_tools.protoc",
    f"-I{proto_dir}",
    f"--python_out={project_root}",
    f"--grpc_python_out={project_root}",
    str(proto_file),
]

print(f"Generating Python code from {proto_file}...")
print(f"Running: {' '.join(cmd)}")

result = subprocess.run(cmd, cwd=project_root)

if result.returncode != 0:
    print("Error: Failed to generate proto code")
    sys.exit(1)

print("✓ Successfully generated Python code from proto file")

# Generate Pydantic models from proto messages
print("\nGenerating Pydantic models from proto messages...")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_pydantic_models",
        project_root / "scripts" / "generate_pydantic_models.py"
    )
    generate_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate_models)
except Exception as e:
    print(f"Warning: Could not generate Pydantic models: {e}")
    print("You can generate them manually by running: python scripts/generate_pydantic_models.py")

