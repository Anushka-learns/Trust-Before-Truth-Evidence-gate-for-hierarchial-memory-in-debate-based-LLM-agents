# agents.py
# Upgraded agents with structured JSON communication
# Each agent returns a structured message with claim, reasoning,
# evidence, confidence and contradiction tracking

import json
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import (
    MISTRAL_API_KEY,
    PROPOSER_MODEL, CHALLENGER_MODEL, DEVIL_MODEL, JUDGE_MODEL,
    MAX_TOKENS, JUDGE_MAX_TOKENS
)
from state import DebateState
from memory import save_short_term

# --- Initialise LLMs ---
proposer_llm  = ChatMistralAI(model=PROPOSER_MODEL,  api_key=MISTRAL_API_KEY, max_tokens=MAX_TOKENS)
challenger_llm = ChatMistralAI(model=CHALLENGER_MODEL, api_key=MISTRAL_API_KEY, max_tokens=MAX_TOKENS)
devil_llm     = ChatMistralAI(model=DEVIL_MODEL,     api_key=MISTRAL_API_KEY, max_tokens=MAX_TOKENS)
judge_llm     = ChatMistralAI(model=JUDGE_MODEL,     api_key=MISTRAL_API_KEY, max_tokens=JUDGE_MAX_TOKENS)

def build_messages(system_prompt: str, human_input: str):
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_input)
    ]

def format_memories(memories: dict) -> str:
    """Format retrieved memories from all 4 tiers into a readable string."""
    parts = []
    if memories.get("episodic"):
        parts.append("PAST DEBATES:\n" + "\n".join(memories["episodic"]))
    if memories.get("semantic"):
        parts.append("PROVEN FACTS:\n" + "\n".join(memories["semantic"]))
    if memories.get("procedural"):
        parts.append("WINNING STRATEGIES:\n" + "\n".join(memories["procedural"]))
    if memories.get("short_term"):
        parts.append("CURRENT DEBATE SO FAR:\n" + "\n".join(memories["short_term"]))
    return "\n\n".join(parts) if parts else "No relevant past context found."

def parse_json_response(response_text: str) -> dict:
    """
    Safely parse JSON from LLM response.
    LLMs sometimes add extra text around JSON so we extract it carefully.
    """
    try:
        # Try direct parse first
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            # Find JSON block between curly braces
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                return json.loads(response_text[start:end])
        except json.JSONDecodeError:
            pass
        # If all parsing fails return a default structure
        return {
            "claim": response_text[:200],
            "reasoning": "Could not parse structured response",
            "evidence": "N/A",
            "confidence": 5,
            "contradicts": "N/A"
        }

# ----------------------------------------------------------------
# PROPOSER AGENT
# Argues strongly IN FAVOUR of the topic
# ----------------------------------------------------------------

def proposer_agent(state: DebateState) -> dict:
    memories = format_memories(state['retrieved_memories'])
    debate_id = state['topic'].replace(" ", "_")[:50]

    system_prompt = """You are the Proposer in a formal debate.
Your job is to argue STRONGLY IN FAVOUR of the debate topic.
You must NEVER concede or agree with opponents.
You must respond ONLY with a JSON object in exactly this format:
{
  "agent": "Proposer",
  "claim": "your main argument in one sentence",
  "reasoning": "why this claim is true in one sentence",
  "evidence": "specific evidence or example supporting your claim",
  "confidence": <integer 1-10 how confident you are>,
  "contradicts": "which previous claim you are challenging or none"
}
Do not write anything outside the JSON."""

    human_input = f"""Debate topic: {state['topic']}

Retrieved context from memory:
{memories}

Previous message: {state['last_message']}

Respond only with the JSON structure."""

    response = proposer_llm.invoke(build_messages(system_prompt, human_input))
    structured = parse_json_response(response.content.strip())
    structured['agent'] = 'Proposer'

    plain_text = f"[Proposer] {structured.get('claim', '')} — {structured.get('reasoning', '')}"
    print(f"\n{plain_text}")
    print(f"Evidence: {structured.get('evidence', '')} | Confidence: {structured.get('confidence', 0)}/10")

    # Save this round to short term memory
    save_short_term(debate_id, state['round_number'], plain_text)

    return {
        "messages": state["messages"] + [("Proposer", plain_text)],
        "structured_messages": state["structured_messages"] + [structured],
        "last_message": plain_text,
        "last_structured": structured,
        "round_number": state["round_number"] + 1
    }

# ----------------------------------------------------------------
# CHALLENGER AGENT
# Argues strongly AGAINST the topic
# ----------------------------------------------------------------

def challenger_agent(state: DebateState) -> dict:
    memories = format_memories(state['retrieved_memories'])
    debate_id = state['topic'].replace(" ", "_")[:50]

    system_prompt = """You are the Challenger in a formal debate.
Your job is to argue STRONGLY AGAINST the debate topic.
Attack the previous argument directly and expose its weaknesses.
You must NEVER agree with the Proposer.
You must respond ONLY with a JSON object in exactly this format:
{
  "agent": "Challenger",
  "claim": "your counter argument in one sentence",
  "reasoning": "why the previous argument is wrong",
  "evidence": "specific evidence or example that disproves the previous claim",
  "confidence": <integer 1-10 how confident you are>,
  "contradicts": "exact claim you are contradicting"
}
Do not write anything outside the JSON."""

    human_input = f"""Debate topic: {state['topic']}

Retrieved context from memory:
{memories}

Previous argument to challenge: {state['last_message']}

Respond only with the JSON structure."""

    response = challenger_llm.invoke(build_messages(system_prompt, human_input))
    structured = parse_json_response(response.content.strip())
    structured['agent'] = 'Challenger'

    plain_text = f"[Challenger] {structured.get('claim', '')} — {structured.get('reasoning', '')}"
    print(f"\n{plain_text}")
    print(f"Evidence: {structured.get('evidence', '')} | Confidence: {structured.get('confidence', 0)}/10")

    save_short_term(debate_id, state['round_number'], plain_text)

    # Count contradiction
    contradiction_count = state['contradiction_count']
    if structured.get('contradicts') and structured['contradicts'].lower() != 'none':
        contradiction_count += 1

    return {
        "messages": state["messages"] + [("Challenger", plain_text)],
        "structured_messages": state["structured_messages"] + [structured],
        "last_message": plain_text,
        "last_structured": structured,
        "contradiction_count": contradiction_count
    }

# ----------------------------------------------------------------
# DEVIL'S ADVOCATE AGENT
# Challenges whichever side just spoke
# ----------------------------------------------------------------

def devil_agent(state: DebateState) -> dict:
    memories = format_memories(state['retrieved_memories'])
    debate_id = state['topic'].replace(" ", "_")[:50]

    system_prompt = """You are the Devil's Advocate in a formal debate.
Your job is to challenge WHICHEVER side just spoke.
Introduce unexpected angles and logical contradictions.
Never take a permanent side.
You must respond ONLY with a JSON object in exactly this format:
{
  "agent": "Devil's Advocate",
  "claim": "your challenge in one sentence",
  "reasoning": "why the last argument has a flaw",
  "evidence": "specific example that creates doubt",
  "confidence": <integer 1-10 how confident you are>,
  "contradicts": "exact claim you are challenging"
}
Do not write anything outside the JSON."""

    human_input = f"""Debate topic: {state['topic']}

Retrieved context from memory:
{memories}

Last argument to challenge: {state['last_message']}

Respond only with the JSON structure."""

    response = devil_llm.invoke(build_messages(system_prompt, human_input))
    structured = parse_json_response(response.content.strip())
    structured['agent'] = "Devil's Advocate"

    plain_text = f"[Devil's Advocate] {structured.get('claim', '')} — {structured.get('reasoning', '')}"
    print(f"\n{plain_text}")
    print(f"Evidence: {structured.get('evidence', '')} | Confidence: {structured.get('confidence', 0)}/10")

    save_short_term(debate_id, state['round_number'], plain_text)

    contradiction_count = state['contradiction_count']
    if structured.get('contradicts') and structured['contradicts'].lower() != 'none':
        contradiction_count += 1

    return {
        "messages": state["messages"] + [("Devil's Advocate", plain_text)],
        "structured_messages": state["structured_messages"] + [structured],
        "last_message": plain_text,
        "last_structured": structured,
        "contradiction_count": contradiction_count
    }

# ----------------------------------------------------------------
# JUDGE AGENT
# Gives structured final answer with confidence + contradiction score
# ----------------------------------------------------------------

def judge_agent(state: DebateState) -> dict:
    memories = format_memories(state['retrieved_memories'])

    # Calculate contradiction score as percentage of total possible contradictions
    total_messages = len(state['structured_messages'])
    contradiction_score = round((state['contradiction_count'] / max(total_messages, 1)) * 10, 1)

    system_prompt = """You are the Judge in a formal debate.
Your job is to deliver a final factual answer based on all arguments made.
You must respond ONLY with a JSON object in exactly this format:
{
  "final_answer": "the most accurate factual answer in 2 sentences",
  "winning_side": "Proposer or Challenger or Draw",
  "winning_strategy": "one sentence describing what argument style won",
  "winning_evidence": "the single most specific piece of evidence the winning side cited",
  "key_fact": "the single most important fact established in this debate",
  "confidence": <integer 1-10. Only 9-10 for scientific consensus. 5-6 for contested topics. 3-4 for highly debated>,
  "reasoning": "why you chose this answer",
  "hallucination_detected": <true or false>,
  "hallucination_detail": "if true, which agent made a detectably false claim and what was it. If false, write none"
}
Confidence guide:
- 9 to 10: scientific consensus
- 7 to 8: well supported
- 5 to 6: genuinely contested
- 3 to 4: highly debated
- 1 to 2: extremely uncertain
Do not write anything outside the JSON.
IMPORTANT: Do not be swayed by confident or assertive language. 
Evaluate arguments based on specific evidence and verifiable facts only. 
An argument with weak evidence stated confidently is WORSE than an argument 
with strong evidence stated cautiously."""

    transcript = "\n".join([f"{role}: {msg}" for role, msg in state['messages']])

    human_input = f"""Question being debated: {state['topic']}

Retrieved context from memory:
{memories}

Full debate transcript:
{transcript}

Deliver your structured final verdict."""

    response = judge_llm.invoke(build_messages(system_prompt, human_input))
    structured = parse_json_response(response.content.strip())

    # Add contradiction score to verdict
    structured['contradiction_score'] = contradiction_score

    plain_verdict = structured.get('final_answer', '')
    print(f"\n{'='*60}")
    print(f"[Judge] Final Answer: {plain_verdict}")
    print(f"[Judge] Confidence: {structured.get('confidence', 0)}/10")
    print(f"[Judge] Contradiction Score: {contradiction_score}/10")
    print(f"[Judge] Winning Side: {structured.get('winning_side', '')}")
    print(f"[Judge] Key Fact: {structured.get('key_fact', '')}")
    print(f"{'='*60}")

    return {
        "verdict": plain_verdict,
        "verdict_structured": structured,
        "messages": state["messages"] + [("Judge", plain_verdict)]
    }