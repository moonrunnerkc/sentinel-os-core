# Author: Bradley R. Kinnard
# utility helpers for sentinel-os-core

import json
import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml
import jsonschema

from config.schemas import system_config_schema, security_rules_schema


logger = logging.getLogger(__name__)


def load_system_config(path: Path | str = "config/system_config.yaml") -> dict[str, Any]:
    """load and validate system config against schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    jsonschema.validate(instance=config, schema=system_config_schema)
    logger.info(f"loaded system config from {path}")
    return config


def load_security_rules(path: Path | str = "config/security_rules.json") -> dict[str, Any]:
    """load and validate security rules against schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"security rules not found: {path}")

    with open(path, "r") as f:
        rules = json.load(f)

    jsonschema.validate(instance=rules, schema=security_rules_schema)
    logger.info(f"loaded security rules from {path}")
    return rules


def compute_hash(data: str | bytes) -> str:
    """compute sha256 hash of data."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """get a configured logger. avoids duplicate handlers."""
    log = logging.getLogger(name)
    log.setLevel(level)

    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)

    return log
