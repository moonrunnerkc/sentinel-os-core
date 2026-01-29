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


def get_hardware_info() -> dict[str, Any]:
    """collect hardware/os info for benchmark reproducibility."""
    import platform
    import os

    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
    }

    # cpu info
    try:
        import psutil
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        info["memory_available_gb"] = round(psutil.virtual_memory().available / (1024**3), 1)
    except ImportError:
        info["cpu_count_logical"] = os.cpu_count()

    # try to get cpu model
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        pass

    # gpu info via nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            info["gpu_name"] = parts[0] if len(parts) > 0 else None
            info["gpu_memory"] = parts[1] if len(parts) > 1 else None
            info["gpu_driver"] = parts[2] if len(parts) > 2 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info
