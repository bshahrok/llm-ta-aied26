"""
JSON output validation and parsing.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple

from config import ALLOWED_CODES


class OutputValidator:
    """Validates and parses model outputs."""

    REQUIRED_KEYS = {"CAD-code", "rationale"}

    @classmethod
    def validate_and_parse(cls, text: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Validates model output and parses JSON.

        Returns:
            Tuple of (is_valid, parsed_dict, error_message)
        """
        text = text.strip()

        # Try parsing the entire string as JSON
        parsed = cls._extract_json(text)
        if parsed is None:
            return False, None, "Could not extract valid JSON from response"

        # Validate structure
        error = cls._validate_structure(parsed)
        if error:
            return False, None, error

        return True, parsed, None

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Attempts to extract and parse JSON from text."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract first {...} block
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    @classmethod
    def _validate_structure(cls, parsed: Dict[str, Any]) -> Optional[str]:
        """Validates the structure and content of parsed JSON."""
        # Check keys
        if set(parsed.keys()) != cls.REQUIRED_KEYS:
            return f"Unexpected keys: {list(parsed.keys())}"

        # Validate code
        code = parsed.get("CAD-code")
        if code not in ALLOWED_CODES:
            return f"Invalid CAD-code: {code}"

        return None