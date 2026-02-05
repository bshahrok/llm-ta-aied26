"""
Configuration settings for CAD coding system.
Centralized configuration for easy debugging and modification.
"""

from dataclasses import dataclass
from typing import FrozenSet


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
CPU_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
GPU_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Generation parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.4
TOP_K = 40

# ============================================================================
# CAD CODEBOOK
# ============================================================================
CAD_CODEBOOK_DICT = {
    "WCT": "The teacher is addressing the whole class.",
    "GT": "The teacher is addressing a group or a student in a group. It also includes any talk: student level",
    "Other": "The teacher isn't talking to the whole class or any groups or students. Either she's silent or talking to herself or a visitor in a non-distracting way",
}

ALLOWED_CODES = frozenset({"WCT", "GT", "Other", "NONE"})

# ============================================================================
# DISCUSSION CONFIGURATION
# ============================================================================
@dataclass
class DiscussionConfig:
    """Configuration for multi-agent discussion behavior."""
    max_rounds: int = 3
    consensus_threshold: float = 0.9
    max_retries_per_agent: int = 2
    allowed_codes: FrozenSet[str] = ALLOWED_CODES

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if not 0 < self.consensus_threshold <= 1:
            raise ValueError("consensus_threshold must be in (0, 1]")
        if self.max_retries_per_agent < 1:
            raise ValueError("max_retries_per_agent must be at least 1")


# ============================================================================
# AGENT ROLES
# ============================================================================
BALANCED_ROLE = "Your job is to weigh evidence, reconcile disagreements, and enforce codebook fidelity."
ADVERSARIAL_ROLE = "Rigorous prosecutor. Be skeptical. Demand direct textual evidence (quote a short phrase). Actively try to falsify other agents' codes. If the text is ambiguous, say so and propose a safe fallback."
CREATIVE_ROLE = "Creative empathic explorer. Look for subtle intent, context, and edge cases. Propose alternative readings and uncommon-but-plausible codes, but justify with text evidence."

# ============================================================================
# OUTPUT SCHEMA
# ============================================================================
EXPECTED_SCHEMA = {
    "CAD-code": "<ONE OF: WCT, GT, Other, NONE>",
    "rationale": "<≤5 sentences, evidence-based>"
}