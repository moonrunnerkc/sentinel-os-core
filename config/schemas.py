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
            "properties": {
                "max_beliefs": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "description": "soft limit for beliefs, null = no limit"
                },
                "max_episodes": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "description": "soft limit for episodes, null = no limit"
                },
                "soft_limit_warning_pct": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "default": 80
                },
                "page_size": {"type": "integer", "minimum": 1, "default": 1000},
                "cache_size": {"type": "integer", "minimum": 1},
                "enable_compression": {"type": "boolean", "default": False}
            }
        },
        "features": {
            "type": "object",
            "required": ["neuromorphic_mode"],
            "properties": {
                "neuromorphic_mode": {"type": "boolean"},
                "zk_proofs": {"type": "boolean"},
                "homomorphic_encryption": {"type": "boolean"}
            }
        },
        "crypto": {
            "type": "object",
            "properties": {
                "commitment_seed": {"type": "integer"}
            }
        }
    }
}

security_rules_schema = {
    "type": "object",
    "required": ["allowed_paths", "hmac_key_seed"],
    "properties": {
        "isolation_level": {
            "type": "string",
            "enum": ["none", "pattern_only", "python_sandbox", "firejail", "docker"],
            "default": "python_sandbox",
            "description": "isolation level - determines threat model and requirements"
        },
        "use_firejail": {
            "type": "boolean",
            "deprecated": True,
            "description": "DEPRECATED: use isolation_level=firejail instead"
        },
        "allowed_paths": {"type": "array", "items": {"type": "string"}},
        "hmac_key_seed": {"type": "integer"},
        "seccomp_profile": {"type": "string"},
        "signature_algorithm": {
            "type": "string",
            "enum": ["ed25519", "dilithium3", "hybrid"],
            "description": "signature algorithm - ed25519 (default), dilithium3 (requires liboqs), hybrid (both)"
        }
    }
}

# meta-evolution schema
meta_evolution_schema = {
    "type": "object",
    "properties": {
        "enabled": {
            "type": "boolean",
            "default": False,
            "description": "enable meta-evolution for hyperparameter optimization"
        },
        "max_generations": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10000,
            "default": 100
        },
        "convergence_threshold": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.001
        },
        "seed": {
            "type": "integer",
            "description": "seed for deterministic evolution"
        },
        "mutation_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.1
        },
        "bounds": {
            "type": "object",
            "properties": {
                "epsilon": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "decay_rate": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "confidence_threshold": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "learning_rate": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2
                }
            }
        }
    }
}

# world model schema
world_model_schema = {
    "type": "object",
    "properties": {
        "enabled": {
            "type": "boolean",
            "default": False,
            "description": "enable world model for causal simulations"
        },
        "type": {
            "type": "string",
            "enum": ["none", "simple"],
            "default": "simple",
            "description": "world model type - simple (numpy-based) or none"
        },
        "decay_rate": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.01
        },
        "regeneration_rate": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.005
        },
        "noise_scale": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.01
        },
        "seed": {
            "type": "integer",
            "description": "seed for deterministic simulation"
        }
    }
}
