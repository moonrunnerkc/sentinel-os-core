# Author: Bradley R. Kinnard
# input layer - input validation and sanitization

import re
from typing import Any

import jsonschema

from utils.helpers import get_logger


logger = get_logger(__name__)


# schema for valid input requests
INPUT_SCHEMA = {
    "type": "object",
    "required": ["type", "content"],
    "properties": {
        "type": {"type": "string", "enum": ["query", "command", "belief", "goal"]},
        "content": {"type": "string", "minLength": 1, "maxLength": 10000},
        "metadata": {"type": "object"},
        "priority": {"type": "number", "minimum": 0, "maximum": 1}
    }
}


# prompt injection patterns to block
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+instructions?",
    r"forget\s+(everything|all|previous)",
    r"you\s+are\s+now\s+a",
    r"pretend\s+(you|to)\s+are",
    r"act\s+as\s+if",
    r"disregard\s+(all|previous)",
    r"override\s+(your|all)\s+instructions?",
    r"new\s+instructions?:",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"\[\s*system\s*\]",
]


class InputLayer:
    """
    input validation and sanitization layer.
    blocks prompt injection attempts.
    """

    def __init__(self):
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
        ]

    def validate(self, input_data: dict[str, Any]) -> tuple[bool, str | None]:
        """validate input against schema."""
        try:
            jsonschema.validate(instance=input_data, schema=INPUT_SCHEMA)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e.message)

    def check_injection(self, text: str) -> tuple[bool, list[str]]:
        """check for prompt injection patterns."""
        matches = []
        for pattern in self._injection_patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)

        is_safe = len(matches) == 0
        if not is_safe:
            logger.warning(f"injection attempt detected: {len(matches)} patterns")

        return is_safe, matches

    def sanitize(self, text: str) -> str:
        """sanitize input text."""
        # remove control characters
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # normalize whitespace
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized

    def process(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """validate, check, and sanitize input."""
        result = {
            "valid": False,
            "safe": False,
            "data": None,
            "errors": []
        }

        # validate schema
        valid, error = self.validate(input_data)
        if not valid:
            result["errors"].append(f"validation: {error}")
            return result

        result["valid"] = True

        # check for injection
        content = input_data.get("content", "")
        safe, patterns = self.check_injection(content)
        if not safe:
            result["errors"].append(f"injection: {patterns}")
            return result

        result["safe"] = True

        # sanitize
        sanitized = input_data.copy()
        sanitized["content"] = self.sanitize(content)
        result["data"] = sanitized

        logger.debug("input processed successfully")
        return result
