"""
Prompt construction for CAD coding agents.
"""

import json
from typing import Any, Dict, Optional

from config import EXPECTED_SCHEMA, ALLOWED_CODES


class PromptBuilder:
    """Handles all prompt construction logic."""

    def __init__(self, name: str, personality: str, role: str, codebook: Dict[str, str], config: Dict[str, Any]):
        self.name = name
        self.personality = personality
        self.role = role
        self.codebook = codebook
        self.config = config

    def build_system_prompt(self, role: str) -> str:
        """Creates the system prompt with instructions and examples."""
        return (
            f"You are {self.name}, a {self.personality} qualitative-coding agent.\n"
            f"Task: {role}.\n\n"
            "CRITICAL: Output ONLY a single JSON object.\n\n"
            "REQUIREMENTS (follow exactly):\n"
            "1) Your ENTIRE response must be ONLY this JSON object and nothing else:\n"
            f"   {json.dumps(EXPECTED_SCHEMA)}\n"
            "2) Use double quotes for JSON strings.\n"
            "3) CAD-code must be one of: WCT, GT, Other, NONE\n"
            "4) Rationale: grounded in evidence from the text.\n"
            "5) If multiple codes could apply, choose the most likely one; if ambiguous, use NONE.\n\n"
            "CORRECT OUTPUT EXAMPLES:\n"
            'Input: "Everybody please listen."\n'
            'Output: {"CAD-code":"WCT","rationale":"Addresses the whole class using \\"Everybody\\" to get attention."}\n\n'
            'Input: "Group 3, read the next paragraph."\n'
            'Output: {"CAD-code":"GT","rationale":"Directs a specific group \\"Group 3\\" to perform an action."}\n\n'
            "Remember: Output ONLY the JSON object. Start your response with { and end with }"
        )

    def build_context_prompt(self) -> str:
        """Creates the codebook context."""
        if not self.codebook:
            return ""
        lines = [f"- {k}: {v}" for k, v in self.codebook.items()]
        return "Codebook:\n" + "\n".join(lines)

    def build_user_prompt(self, text: str) -> str:
        """Creates the user prompt with the text to annotate."""
        template = self.config.get(
            "user_template",
            'text to code: \n{text}\n\n'
        )
        return template.format(text=text)

    def build_full_prompt(
        self,
        text: str,
        role: str,
        extra_context: Optional[str] = None,
        previous_turn: Optional[str] = None
    ) -> Dict[str, str]:
        """Builds complete prompt dictionary with all components."""
        prompt = {
            "system": self.build_system_prompt(role),
            "context": self.build_context_prompt(),
            "user": self.build_user_prompt(text),
        }

        if extra_context:
            prompt["extra"] = extra_context

        if previous_turn:
            # Append previous turn to context
            if prompt.get("context"):
                prompt["context"] += f"\n\n###\nPrevious turn: {previous_turn}"
            else:
                prompt["user"] += f"\n\n###\nPrevious turn: {previous_turn}"

        return prompt

    def build_retry_prompt(self, original_prompt: str, failed_output: str) -> str:
        """Builds a retry prompt when the model fails to produce valid JSON."""
        return (
            f"{original_prompt}\n\n"
            "--- RETRY REQUEST ---\n"
            "Your previous output was invalid or incorrectly formatted.\n"
            f"Previous output:\n{failed_output[:500]}\n\n"
            "Please output ONLY a valid JSON object with this exact structure:\n"
            f"{json.dumps(EXPECTED_SCHEMA)}\n\n"
            "Requirements:\n"
            "- Start with { and end with }\n"
            "- No markdown, no explanations, no extra text\n"
            f"- CAD-code must be exactly one of: {', '.join(sorted(ALLOWED_CODES))}\n"
            "- Use double quotes for strings\n\n"
            "Return ONLY the JSON object now:"
        )

    @staticmethod
    def to_string(prompt_dict: Dict[str, str]) -> str:
        """Converts prompt dictionary to a formatted string."""
        parts = []
        for key in ["system", "context", "extra", "user"]:
            if prompt_dict.get(key):
                parts.append(f"=== {key.upper()} ===\n{prompt_dict[key]}")
        return "\n\n".join(parts)