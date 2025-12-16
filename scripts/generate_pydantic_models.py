#!/usr/bin/env python3
"""
Script to generate Pydantic models from Protocol Buffer messages.

This script uses protobuf-to-pydantic to convert proto message classes
to Pydantic BaseModel classes, eliminating the need for manual model definitions.

Usage:
    python scripts/generate_pydantic_models.py

Note: This script requires that proto code be generated first:
    python scripts/generate_proto.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from protobuf_to_pydantic import msg_to_pydantic_model
except ImportError:
    print("Error: protobuf-to-pydantic not installed.")
    print("Install it with: pip install protobuf-to-pydantic")
    sys.exit(1)

try:
    from gym_mcp_server.proto import gym_service_pb2
except ImportError:
    print("Error: Proto code not generated.")
    print("Run: python scripts/generate_proto.py")
    sys.exit(1)

# Output file - place in proto folder alongside other generated code
models_file = project_root / "gym_mcp_server" / "proto" / "models.py"

print("Generating Pydantic models from proto messages using protobuf-to-pydantic...")

# Start with header
models_code = '''"""
Pydantic models generated from Protocol Buffer definitions.

This file is auto-generated from gym_service.proto using protobuf-to-pydantic.
DO NOT EDIT MANUALLY. To regenerate, run:
    python scripts/generate_pydantic_models.py

The models are derived from:
    gym_mcp_server/proto/gym_service.proto

These models are adapted for HTTP/FastAPI use, using simpler types
(Any for observations/actions) compared to the structured proto types.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


'''

# Map proto messages to model names
proto_model_map = {
    "ResetRequest": gym_service_pb2.ResetRequest,
    "ResetResponse": gym_service_pb2.ResetResponse,
    "StepRequest": gym_service_pb2.StepRequest,
    "StepResponse": gym_service_pb2.StepResponse,
    "RenderRequest": gym_service_pb2.RenderRequest,
    "RenderResponse": gym_service_pb2.RenderResponse,
    "GetInfoRequest": gym_service_pb2.GetInfoRequest,
    "GetInfoResponse": gym_service_pb2.GetInfoResponse,
    "CloseRequest": gym_service_pb2.CloseRequest,
    "CloseResponse": gym_service_pb2.CloseResponse,
    "HealthRequest": gym_service_pb2.HealthRequest,
    "HealthResponse": gym_service_pb2.HealthResponse,
}

# Generate models using protobuf-to-pydantic
for model_name, proto_class in proto_model_map.items():
    try:
        # Use protobuf-to-pydantic to generate the model
        pydantic_model = msg_to_pydantic_model(proto_class)
        
        # Get field information from the generated model
        fields = []
        for field_name, field_info in pydantic_model.model_fields.items():
            # Get the annotation as a string
            annotation_str = str(field_info.annotation)
            # Clean up the annotation string
            annotation_str = annotation_str.replace("typing.", "").replace("builtins.", "")
            
            # Determine default value
            if field_info.is_required() and field_info.default is ...:
                default = "Field(...)"
            elif field_info.default is not ...:
                if field_info.default is None:
                    default = "Field(default=None)"
                elif isinstance(field_info.default, (list, dict)):
                    default = f"Field(default_factory={type(field_info.default).__name__})"
                else:
                    default = f"Field(default={repr(field_info.default)})"
            else:
                default = "Field(...)"
            
            # Add description if available
            description = getattr(field_info, 'description', None) or ""
            if description:
                default = default.replace(")", f', description="{description}")')
            
            fields.append(f'    {field_name}: {annotation_str} = {default}')
        
        # Create model class
        models_code += f'''class {model_name}(BaseModel):
    """Pydantic model for {proto_class.DESCRIPTOR.name} (generated from proto)."""
    
'''
        if fields:
            models_code += "\n".join(fields)
        else:
            models_code += "    pass"
        models_code += "\n\n"
        
        print(f"  ✓ Generated {model_name}")
        
    except Exception as e:
        print(f"  ✗ Warning: Could not auto-generate {model_name}: {e}")
        # Fallback: create basic model from proto descriptor
        fields = []
        for field_desc in proto_class.DESCRIPTOR.fields:
            field_name = field_desc.name
            
            # Map proto types to Python types
            if field_desc.type == field_desc.TYPE_INT64 or field_desc.type == field_desc.TYPE_INT32:
                py_type = "int"
            elif field_desc.type == field_desc.TYPE_DOUBLE or field_desc.type == field_desc.TYPE_FLOAT:
                py_type = "float"
            elif field_desc.type == field_desc.TYPE_BOOL:
                py_type = "bool"
            elif field_desc.type == field_desc.TYPE_STRING:
                py_type = "str"
            elif field_desc.type == field_desc.TYPE_MESSAGE:
                # For nested messages (Observation, Action, EnvInfo), use Any for HTTP compatibility
                py_type = "Any"
            else:
                py_type = "Any"
            
            # Handle optional and repeated fields
            if field_desc.label == field_desc.LABEL_OPTIONAL:
                py_type = f"Optional[{py_type}]"
                default = 'Field(default=None)'
            elif field_desc.label == field_desc.LABEL_REPEATED:
                py_type = f"List[{py_type}]"
                default = 'Field(default_factory=list)'
            else:
                default = 'Field(...)'
            
            fields.append(f'    {field_name}: {py_type} = {default}')
        
        models_code += f'''class {model_name}(BaseModel):
    """Pydantic model for {proto_class.DESCRIPTOR.name} (generated from proto descriptor)."""
    
'''
        if fields:
            models_code += "\n".join(fields)
        else:
            models_code += "    pass"
        models_code += "\n\n"

# Add EnvInfoResponse as a wrapper for HTTP API compatibility
models_code += '''class EnvInfoResponse(BaseModel):
    """Response model for environment info (HTTP API wrapper)."""

    env_info: Dict[str, Any] = Field(description="Environment metadata")
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")

'''

# Write the models file
models_file.write_text(models_code)
print(f"\n✓ Successfully generated Pydantic models in {models_file}")
