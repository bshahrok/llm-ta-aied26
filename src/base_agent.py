"""
Base coding agent with LLM-based text annotation.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, Optional

from config import ALLOWED_CODES
from validator import OutputValidator
from prompt_builder import PromptBuilder
from model_manager import MockModelManager, ModelManager
from utils.agent import heuristic_label

logger = logging.getLogger(__name__)


class BaseCodingAgent:
    """Base class for coding agents with LLM-based text annotation."""

    DEFAULT_TEMPERATURE = 0.4

    def __init__(
        self,
        name: str,
        personality: str,
        role: str,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = 40,
        codebook: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        use_heuristic: bool = True,
        use_mock_llm: bool = False,
        mock_response: Optional[str] = None,
        mock_generator: Optional[Callable[[str], str]] = None,
    ):
        self.name = name
        self.personality = personality
        self.role = role
        self.debug = debug
        self.use_heuristic = use_heuristic

        # Initialize components
        self.validator = OutputValidator()
        self.prompt_builder = PromptBuilder(
            name, personality, role,
            codebook or {}, config or {}
        )
        if use_mock_llm:
            self.model_manager = MockModelManager(
                response=mock_response,
                generator=mock_generator
            )
        else:
            self.model_manager = ModelManager(
                model_id, device, temperature, top_k
            )

        # Store options
        self.options = {"temperature": temperature, "top_k": top_k}
        self.codebook = codebook or {}
        self.config = config or {}

    def chat(self, text: str, max_retries: int, role: Optional[str] = None, 
             extra_context: Optional[str] = None, **gen_opts):
        """Main chat interface for the agent to interact with the language model."""
        # Build prompt
        prompt_str = self.get_prompt_str(
            text=text,
            role=role or self.role,
            extra_context=extra_context,
            **gen_opts
        )

        # Log if debug enabled
        if self.debug:
            logger.info("=== PROMPT ===")
            logger.info(prompt_str)

        # Generate response
        return self._call_and_retry(prompt_str, max_retries, **gen_opts)

    def _call_and_retry(
        self,
        prompt_str: str,
        max_retries: int,
        **gen_opts
    ) -> str:
        """Internal method to handle generation with retries."""
        try:
            raw = self.model_manager.generate(prompt_str, **gen_opts)
            valid, parsed, err = self.validator.validate_and_parse(raw)

            if valid:
                return raw

            logger.debug("Initial attempt failed: %s. Raw: %s", err, raw[:500])

            # Retry with structured prompt
            for attempt in range(max_retries):
                retry_prompt = self.prompt_builder.build_retry_prompt(
                    prompt_str, raw
                )
                logger.debug("Retry attempt %d/%d", attempt + 1, max_retries)

                raw = self.model_manager.generate(retry_prompt, **gen_opts)
                valid, parsed, err = self.validator.validate_and_parse(raw)

                if valid:
                    return raw

                logger.debug(
                    "Retry %d failed: %s. Raw: %s",
                    attempt + 1, err, raw[:500]
                )
            if self.use_heuristic:
                # Fallback to heuristic
                logger.warning(
                    "Model failed after %d retries. Using heuristic fallback.",
                    max_retries
                )
                return json.dumps(heuristic_label(prompt_str))

        except Exception as e:
            logger.error("Error during generation: %s", e, exc_info=True)
            return json.dumps({
                "CAD-code": "NONE",
                "rationale": f"Error: {str(e)}"
            })

    def get_prompt_str(self, text: str, role: Optional[str] = None,
                      extra_context: Optional[str] = None, **gen_opts):
        """Build and return prompt string."""
        prompt_dict = self.prompt_builder.build_full_prompt(
            text=text,
            role=role or self.role,
            extra_context=extra_context,
            previous_turn=None,
            **gen_opts
        )

        return self.prompt_builder.to_string(prompt_dict)

    def get_agent_info(self) -> Dict[str, Any]:
        """Return agent configuration as dictionary."""
        return {
            "name": self.name,
            "personality": self.personality,
            "role": self.role,
            "model": self.model_manager.model_id,
            "device": str(self.model_manager.device),
            "options": self.options,
            "codebook": self.codebook,
            "config": self.config,
            "debug": self.debug,
        }

    def validate_and_parse(self, text: str):
        """Validate and parse response."""
        return self.validator.validate_and_parse(text)

    def get_parsed_resp(self, text: str):
        """Parse response."""
        valid, parsed, err = self.validator.validate_and_parse(text)
        return parsed if valid else None


class SingleAgentCoding(BaseCodingAgent):
    """Single agent coding implementation."""
    
    def assign_code(self, text: str, max_retries: int = 2,
                    extra_context: Optional[str] = None,
                    **gen_opts) -> str:
        """Assigns a code to a given text based on the codebook and generates a rationale."""
        logger.debug(f"Assigning code for text: {text}")
        
        response = self.chat(
            text=text,
            role=self.role,
            max_retries=max_retries,
            extra_context=extra_context,
            **gen_opts
        )
        
        logger.debug(f"Raw response from agent: {response}")
        return response