# Author: Bradley R. Kinnard
# output layer - output formatting and security filtering

import re
from typing import Any

from utils.helpers import get_logger


logger = get_logger(__name__)


# patterns to filter from output
FILTER_PATTERNS = [
    r"password\s*[:=]\s*\S+",
    r"api[_-]?key\s*[:=]\s*\S+",
    r"secret\s*[:=]\s*\S+",
    r"token\s*[:=]\s*\S+",
    r"-----BEGIN\s+\w+\s+PRIVATE\s+KEY-----",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # emails
]


class OutputLayer:
    """
    output formatting and security filtering.
    removes sensitive data before returning.
    """

    def __init__(self):
        self._filter_patterns = [
            re.compile(p, re.IGNORECASE) for p in FILTER_PATTERNS
        ]

    def filter_sensitive(self, text: str) -> str:
        """remove sensitive data from text."""
        filtered = text
        for pattern in self._filter_patterns:
            filtered = pattern.sub("[REDACTED]", filtered)
        return filtered

    def format_response(
        self,
        content: Any,
        response_type: str = "text",
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """format response with type and metadata."""
        if isinstance(content, str):
            content = self.filter_sensitive(content)

        response = {
            "type": response_type,
            "content": content,
            "metadata": metadata or {},
            "filtered": True
        }

        logger.debug(f"formatted {response_type} response")
        return response

    def format_error(
        self,
        error: str | Exception,
        code: str = "ERROR"
    ) -> dict[str, Any]:
        """format error response."""
        message = str(error)
        # filter sensitive data from errors too
        message = self.filter_sensitive(message)

        return {
            "type": "error",
            "code": code,
            "message": message,
            "filtered": True
        }

    def format_belief(self, belief: dict[str, Any]) -> dict[str, Any]:
        """format belief for output."""
        # filter content
        if "content" in belief:
            belief["content"] = self.filter_sensitive(str(belief["content"]))

        return {
            "type": "belief",
            "content": belief,
            "filtered": True
        }

    def format_goal(self, goal: dict[str, Any]) -> dict[str, Any]:
        """format goal for output."""
        if "description" in goal:
            goal["description"] = self.filter_sensitive(str(goal["description"]))

        return {
            "type": "goal",
            "content": goal,
            "filtered": True
        }

    def truncate(self, text: str, max_length: int = 1000) -> str:
        """truncate text to max length."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
