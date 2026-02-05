"""
Data structures for multi-agent discussions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentResponse:
    """Represents a single agent's response in a discussion round.

    Attributes:
        agent: Name/identifier of the agent
        code: CAD code assigned (e.g., 'WCT', 'GT', 'Other', 'NONE')
        rationale: Agent's reasoning for the code choice
        raw: Raw unparsed response from the agent
        round: Round number (1-indexed)
    """
    agent: str
    code: str
    rationale: str
    raw: str
    round: int

    def __post_init__(self):
        """Validate response data."""
        if self.round < 1:
            raise ValueError(f"Round must be >= 1, got {self.round}")
        if not self.agent:
            raise ValueError("Agent name cannot be empty")

    def convert_to_dict(self) -> Dict[str, Any]:
        """Convert response to a dictionary."""
        return {
            "agent": self.agent,
            "code": self.code,
            "rationale": self.rationale,
            "raw": self.raw,
            "round": self.round,
        }


@dataclass
class DiscussionResult:
    """Results from a multi-agent discussion process.

    Attributes:
        final_code: Consensus or plurality code
        final_rationale: Combined rationale from agents who chose final_code
        confidence: Agreement ratio (0.0 to 1.0)
        history: List of responses per round
        tallies: Vote counts per round
        consensus_reached: Whether consensus threshold was met
        num_rounds: Total number of rounds conducted
    """
    text_to_code: str
    human_code: str
    final_code: str
    final_rationale: str
    confidence: float
    history: List[List[AgentResponse]] = field(default_factory=list)
    round_dicts: List[Dict[str, Any]] = field(default_factory=list)
    tallies: List[Dict[str, int]] = field(default_factory=list)
    consensus_reached: bool = False
    num_rounds: int = 0
    num_agents: int = 0

    def __post_init__(self):
        """Validate result data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.num_rounds < 0:
            raise ValueError(f"num_rounds must be >= 0, got {self.num_rounds}")

    def get_num_agents(self) -> int:
        """Get number of participating agents."""
        if self.history:
            return len(self.history[0])
        return 0

    def get_round_dicts(self) -> List[Dict[str, Any]]:
        """Convert history to a list of dictionaries."""
        res = []
        for round_idx, (responses, tally) in enumerate(zip(self.history, self.tallies), 1):
            round_dict = {
                "round_num": round_idx,
                "votes": dict(tally),
                "responses": [resp.convert_to_dict() for resp in responses]
            }
            res.append(round_dict)
        self.round_dicts = res
        return res

    def get_agent_journey(self, agent_name: str) -> List[AgentResponse]:
        """Track how a specific agent voted across rounds."""
        journey = []
        for round_responses in self.history:
            for resp in round_responses:
                if resp.agent == agent_name:
                    journey.append(resp)
                    break
        return journey