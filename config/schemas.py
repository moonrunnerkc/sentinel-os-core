# Author: Bradley R. Kinnard
# schema definitions for zero-trust config validation

system_config_schema = {
    "type": "object",
    "required": ["llm", "performance", "features"],
    "properties": {
        "llm": {
            "type": "object",
            "required": ["backend", "model_path", "temperature", "seed"],
            "properties": {
                "backend": {"type": "string", "enum": ["llama-cpp", "onnx"]},
                "model_path": {"type": "string"},
                "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                "seed": {"type": "integer"},
                "gpu_layers": {"type": "integer", "minimum": 0},
                "min_vram_gb": {"type": "number", "minimum": 0.0}
            }
        },
        "performance": {
            "type": "object",
            "required": ["max_beliefs", "max_episodes"],
            "properties": {
                "max_beliefs": {"type": "integer", "minimum": 1},
                "max_episodes": {"type": "integer", "minimum": 1},
                "cache_size": {"type": "integer", "minimum": 1}
            }
        },
        "features": {
            "type": "object",
            "required": ["neuromorphic_mode"],
            "properties": {
                "neuromorphic_mode": {"type": "boolean"}
            }
        }
    }
}

security_rules_schema = {
    "type": "object",
    "required": ["use_firejail", "allowed_paths", "hmac_key_seed"],
    "properties": {
        "use_firejail": {"type": "boolean"},
        "allowed_paths": {"type": "array", "items": {"type": "string"}},
        "hmac_key_seed": {"type": "integer"},
        "seccomp_profile": {"type": "string"}
    }
}
