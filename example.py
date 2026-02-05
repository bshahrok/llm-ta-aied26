"""
Example runner for CAD coding system.
Simple, debuggable demonstration of the multi-agent discussion system.
"""

import json
import logging
import sys
# Setup logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from config import (
    CAD_CODEBOOK_DICT,
    DiscussionConfig,
    BALANCED_ROLE,
)

from utils.agent import heuristic_label
from src.base_agent import SingleAgentCoding
from src.MAD import MultiAgentDiscussion


def _mock_generator_as_str(text: str) -> str:
    """Adapt heuristic_label's dict output to valid JSON."""
    return json.dumps(heuristic_label(text))


def main():
    """Run a simple example of the CAD coding system."""
    
    # Example text to code
    text = "So remember you guys are in groups so talk to your partner about the cards you move. Make sure your partner agrees with you."
    
    print("="*80)
    print("CAD Coding System - Multi-Agent Discussion Example")
    print("="*80)
    print(f"\nText to code:\n{text}\n")
    
    # Create three agents with different personalities
    print("Creating agents...")
    agents = {
        "ava": SingleAgentCoding(
            "Ava",
            "balanced arbiter",
            BALANCED_ROLE,
            debug=False,
            codebook=CAD_CODEBOOK_DICT,
            use_mock_llm=True,
            mock_generator=_mock_generator_as_str
        )
        # "ben": SingleAgentCoding(
        #     "Ben",
        #     "rigorous and concise",
        #     ADVERSARIAL_ROLE,
        #     debug=False,
        #     codebook=CAD_CODEBOOK_DICT
        # ),
        # "cam": SingleAgentCoding(
        #     "Cam",
        #     "creative and empathic",
        #     CREATIVE_ROLE,
        #     debug=False,
        #     codebook=CAD_CODEBOOK_DICT
        # )
    }
    
    # Configure discussion
    config = DiscussionConfig(
        max_rounds=1,
        consensus_threshold=0.9,
        max_retries_per_agent=1,
    )
    print("Discussion configuration set.")
    
    # Create multi-agent discussion
    print("Initializing multi-agent discussion...")
    mad = MultiAgentDiscussion(
        list(agents.values()),
        config=config
    )
    
    # Run discussion
    print("\nStarting discussion...\n")
    result = mad.discuss(text, human_code="GT")
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Final Code: {result.final_code}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Consensus Reached: {result.consensus_reached}")
    print(f"Rounds: {result.num_rounds}")
    print(f"\nFinal Rationale:\n{result.final_rationale}")
    
    # Show round-by-round breakdown
    print("\n" + "="*80)
    print("ROUND-BY-ROUND BREAKDOWN")
    print("="*80)
    for round_idx, (responses, tally) in enumerate(zip(result.history, result.tallies), 1):
        print(f"\nRound {round_idx}:")
        print(f"Votes: {dict(tally)}")
        for resp in responses:
            print(f"  • {resp.agent}: {resp.code}")
            if resp.rationale:
                print(f"    → {resp.rationale[:100]}...")
    
    print("\n" + "="*80)
    print("Example completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()