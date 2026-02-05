import re
from typing import Dict


def heuristic_label(text: str) -> Dict[str, str]:
        """Fallback heuristic labeling when model fails."""
        lower = (text or "").lower().strip()

        if not lower:
            return {
                "CAD-code": "NONE",
                "rationale": "Empty input"
            }

        # Check patterns in priority order
        patterns = [
            (r'\b(everybody|everyone|class|students|all of you|all)\b',
             "WCT", "whole-class addressing"),
            (r'\b(group|pair|you two|you three)\b',
             "GT", "group-level addressing"),
            (r'^[A-Z][a-z]+,', "GT", "direct student address"),
        ]

        for pattern, code, reason in patterns:
            if re.search(pattern, text):
                return {"CAD-code": code, "rationale": reason}

        return {
            "CAD-code": "Other",
            "rationale": "Non-directed teacher talk"
        }