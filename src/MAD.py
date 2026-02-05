"""
Multi-agent discussion system for collaborative coding.
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from config import DiscussionConfig
from discussion_structures import AgentResponse, DiscussionResult

logger = logging.getLogger(__name__)


class MultiAgentDiscussion:
    """Multi-agent discussion system with consensus-based decision making."""

    def __init__(
        self,
        agents: List[Any],
        config: Optional[DiscussionConfig] = None
    ):
        # Validate inputs
        if not agents:
            raise ValueError("At least one agent is required")

        # Initialize configuration
        self.config = config or DiscussionConfig()
        self.agents = agents
        self.round_num = 0
        self.history = []
        self.tallies = []
        self.consensus_reached = False
        self.logger = logging.getLogger(self.__class__.__name__)

    def _to_agent_response(self, agent: Any, raw: str, round_num: int) -> AgentResponse:
        """Convert raw agent output into AgentResponse."""
        valid, parsed, err = agent.validate_and_parse(raw)

        if not valid or not isinstance(parsed, dict):
            code = "NONE"
            rationale = f"Parse error/invalid format: {err}"
        else:
            code = parsed.get("CAD-code", "NONE")
            rationale = parsed.get("rationale", "") or ""

        # Validate code
        if code not in self.config.allowed_codes:
            self.logger.warning(f"Invalid code received: {code}")
            code = "NONE"

        return AgentResponse(
            agent=getattr(agent, "name", str(agent)),
            code=code,
            rationale=rationale,
            raw=raw,
            round=round_num,
        )

    def _tally_round(self, round_responses: List[AgentResponse]) -> Dict[str, int]:
        """Count votes for a single round."""
        return dict(Counter(r.code for r in round_responses))

    def _build_discussion_context(
        self,
        prev_responses: Optional[List[AgentResponse]],
        round_num: int
    ) -> str:
        """Build human-readable discussion context from previous round."""
        if not prev_responses:
            return f"Round 1: Provide your independent assessment."

        lines = [f"Round {round_num}: Previous round responses:", ""]

        # Show each agent's position and rationale
        for resp in prev_responses:
            rationale = resp.rationale if resp.rationale else "(no rationale provided)"
            lines.append(f"- {resp.agent} chose '{resp.code}': {rationale}")

        lines.append("")

        # Add vote distribution summary
        codes = [r.code for r in prev_responses if r.code != "NONE"]
        if codes:
            tally = Counter(codes)
            tally_str = ", ".join(f"{code}: {count}" for code, count in tally.most_common())
            lines.append(f"Vote distribution: {tally_str}")

        lines.extend([
            "",
            "Consider the above responses. You may:",
            "- Change your assessment if you find others' reasoning convincing",
            "- Maintain your position if you believe your reasoning is stronger",
            "- Provide additional rationale to explain your choice"
        ])

        return "\n".join(lines)

    def is_consensus_reached(
        self,
        tally: Dict[str, int],
        threshold: float = 0.8
    ) -> Tuple[Optional[str], float]:
        """Determines if consensus is reached among agents."""
        # Ignore NONE votes
        valid_votes = {k: v for k, v in tally.items() if k != "NONE"}
        
        if not valid_votes:
            return None, 0.0

        total_n = sum(valid_votes.values())
        top_code, top_count = Counter(valid_votes).most_common(1)[0]
        
        agreement_ratio = top_count / total_n

        if agreement_ratio >= threshold:
            self.logger.info(f"Consensus reached on code '{top_code}' with agreement {agreement_ratio:.2f}")
            return top_code, agreement_ratio

        self.logger.info(f"No consensus: top code '{top_code}' has agreement {agreement_ratio:.2f}")
        return None, agreement_ratio

    def _finalize(self, last_round_responses: List[AgentResponse]) -> Tuple[str, str, float]:
        """Finalize the discussion by plurality vote."""
        codes = [resp.code for resp in last_round_responses]
        counts = Counter(codes)
        
        if not counts:
            return "NONE", "", 0.0

        final_code, freq = counts.most_common(1)[0]
        confidence = freq / len(self.agents)

        # Aggregate rationales for the final code
        rationales = [
            resp.rationale for resp in last_round_responses
            if resp.code == final_code and resp.rationale
        ]
        final_rationale = " | ".join(rationales)

        return final_code, final_rationale, confidence

    def reset(self) -> None:
        """Reset discussion state for reuse."""
        self.round_num = 0
        self.history.clear()
        self.tallies.clear()
        self.consensus_reached = False
        self.logger.debug("Discussion state reset")

    def discuss(self, text: str, human_code: str = "", **kwargs) -> DiscussionResult:
        """Run multi-agent discussion with consensus detection."""
        self.reset()
        self.logger.info("== Starting MultiAgentDiscussion with %d agents", len(self.agents))

        for round_idx in range(self.config.max_rounds):
            self.round_num = round_idx + 1
            self.logger.info("Round %d/%d", self.round_num, self.config.max_rounds)

            # Build context from previous round
            prev_resp = self.history[-1] if self.history else None
            ctx = self._build_discussion_context(prev_resp, self.round_num)

            # Collect responses for this round
            round_responses: List[AgentResponse] = []
            for agent in self.agents:
                max_retries = self.config.max_retries_per_agent

                raw = agent.assign_code(
                    text,
                    extra_context=ctx,
                    max_retries=max_retries,
                    **kwargs
                )
                round_responses.append(self._to_agent_response(agent, raw, self.round_num))

            # Save round artifacts
            self.history.append(round_responses)
            tally = self._tally_round(round_responses)
            self.tallies.append(tally)
            self.logger.debug(f"Round {self.round_num} tally: {tally}")

            # Check for consensus
            consensus_code, agreement = self.is_consensus_reached(
                tally,
                self.config.consensus_threshold
            )
            
            if consensus_code:
                self.consensus_reached = True
                rationales = [
                    resp.rationale for resp in round_responses
                    if resp.code == consensus_code and resp.rationale
                ]

                return DiscussionResult(
                    text_to_code=text,
                    human_code=human_code,
                    final_code=consensus_code,
                    final_rationale=" | ".join(rationales),
                    confidence=agreement,
                    history=self.history,
                    tallies=self.tallies,
                    consensus_reached=True,
                    num_rounds=self.round_num
                )

        # Max rounds reached without consensus - finalize by plurality
        final_code, final_rationale, confidence = self._finalize(self.history[-1])

        return DiscussionResult(
            text_to_code=text,
            human_code=human_code,
            final_code=final_code,
            final_rationale=final_rationale,
            confidence=confidence,
            history=self.history,
            tallies=self.tallies,
            consensus_reached=False,
            num_rounds=self.round_num
        )