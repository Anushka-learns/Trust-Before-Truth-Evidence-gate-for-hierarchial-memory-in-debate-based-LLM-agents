# state.py

# Upgraded state with structured messages, hierarchical memory,
# confidence scoring and contradiction scoring

from typing import TypedDict, List, Dict, Any

class DebateState(TypedDict):
    # The factual question being debated
    topic: str

    # Plain text messages for display
    # (role, plain_text) tuples
    messages: List[tuple]

    # Structured JSON messages from each agent
    # Contains claim, reasoning, evidence, confidence
    structured_messages: List[Dict[str, Any]]

    # Current round number
    round_number: int

    # Last plain text message for next agent to respond to
    last_message: str

    # Last structured message for next agent to reference
    last_structured: Dict[str, Any]

    # Judge's final plain text answer
    verdict: str

    # Judge's structured output
    # Contains final_answer, confidence, contradiction_score, key_fact
    verdict_structured: Dict[str, Any]

    # Retrieved context from all 4 memory tiers
    # short_term, episodic, semantic, procedural
    retrieved_memories: Dict[str, List[str]]

    # Tracks contradiction count during debate
    contradiction_count: int