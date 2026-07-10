# memory.py
# Falsifiable Hierarchical Memory System
# 4 tiers: short-term, episodic, semantic, procedural
# Semantic facts have trust scores, provenance, contradiction detection
# Procedural memory only saves evidence-backed wins

import re
import json
import chromadb
from datetime import datetime
from sentence_transformers import SentenceTransformer
from config import CHROMA_DB_PATH, EMBEDDING_MODEL, TOP_K_MEMORIES

# --- Initialise ---
embedder = SentenceTransformer(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

short_term  = client.get_or_create_collection(name="short_term_memory")
episodic    = client.get_or_create_collection(name="episodic_memory")
semantic    = client.get_or_create_collection(name="semantic_memory")
procedural  = client.get_or_create_collection(name="procedural_memory")

# ----------------------------------------------------------------
# SHORT TERM MEMORY — unchanged
# ----------------------------------------------------------------

def save_short_term(debate_id: str, round_num: int, content: str):
    embedding = embedder.encode(content).tolist()
    doc_id = f"{debate_id}_round_{round_num}"
    short_term.upsert(
        documents=[content],
        embeddings=[embedding],
        ids=[doc_id],
        metadatas=[{"debate_id": debate_id, "round": round_num}]
    )

def get_short_term(debate_id: str) -> list:
    if short_term.count() == 0:
        return []
    results = short_term.get(where={"debate_id": debate_id})
    return results['documents'] if results['documents'] else []

def clear_short_term(debate_id: str):
    results = short_term.get(where={"debate_id": debate_id})
    if results['ids']:
        short_term.delete(ids=results['ids'])
    print(f"[Memory] Short-term cleared for: {debate_id[:30]}")

# ----------------------------------------------------------------
# EPISODIC MEMORY — with threshold
# ----------------------------------------------------------------

def save_episodic(question: str, transcript: str, verdict: str, confidence: float, threshold: float = 7.0):
    if confidence < threshold:
        print(f"[Episodic] SKIPPED — confidence {confidence} below threshold {threshold}")
        return False

    full_text = f"Question: {question}\n\nTranscript:\n{transcript}\n\nFinal Answer:\n{verdict}"
    embedding = embedder.encode(full_text).tolist()
    debate_id = question.replace(" ", "_")[:50]

    episodic.upsert(
        documents=[full_text],
        embeddings=[embedding],
        ids=[debate_id],
        metadatas=[{
            "question": question,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }]
    )
    print(f"[Episodic] Saved — confidence {confidence}")
    return True

def retrieve_episodic(question: str) -> list:
    if episodic.count() == 0:
        return []
    query_embedding = embedder.encode(question).tolist()
    results = episodic.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K_MEMORIES, episodic.count()),
        include=["documents", "distances"]
    )
    memories = []
    if results['documents']:
        for doc, distance in zip(results['documents'][0], results['distances'][0]):
            if distance < 1.0:
                memories.append(doc)
    return memories

# ----------------------------------------------------------------
# SEMANTIC MEMORY — fully falsifiable
# Trust score, provenance, contradiction detection
# ----------------------------------------------------------------

def is_contradicting(new_fact: str, existing_fact: str) -> bool:
    """
    Simple contradiction detection.
    Checks if new fact and existing fact are semantically similar
    but contain opposing signals.
    """
    contradiction_pairs = [
        ("true", "false"), ("correct", "incorrect"),
        ("did", "did not"), ("is", "is not"),
        ("was", "was not"), ("happened", "never happened"),
        ("real", "fake"), ("exists", "does not exist"),
        ("proved", "disproved"), ("confirmed", "denied")
    ]
    new_lower = new_fact.lower()
    existing_lower = existing_fact.lower()

    for word1, word2 in contradiction_pairs:
        if (word1 in new_lower and word2 in existing_lower) or \
           (word2 in new_lower and word1 in existing_lower):
            return True
    return False

def save_semantic(fact: str, confidence: float, source_question: str, debate_id: str):
    """
    Saves fact to semantic memory with full provenance.
    Checks for contradictions before writing.
    Updates trust score if fact already exists.
    """
    embedding = embedder.encode(fact).tolist()
    fact_id = fact.replace(" ", "_")[:50]
    timestamp = datetime.now().isoformat()

    # Step 1 — check if similar fact already exists
    if semantic.count() > 0:
        results = semantic.query(
            query_embeddings=[embedding],
            n_results=min(3, semantic.count()),
            include=["documents", "distances", "metadatas"]
        )

        if results['documents'] and results['documents'][0]:
            for existing_doc, distance, existing_meta in zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            ):
                # Very similar fact found
                if distance < 0.3:
                    existing_trust = existing_meta.get('trust_score', 5.0)
                    existing_confirmed = existing_meta.get('confirmed_count', 1)

                    # Check if they contradict each other
                    if is_contradicting(fact, existing_doc):
                        # Contradiction found — decrement trust of existing fact
                        new_trust = max(0, existing_trust - 2)
                        existing_id = existing_meta.get('fact_id', fact_id)
                        print(f"[Semantic] CONTRADICTION detected — trust score: {existing_trust} → {new_trust}")

                        if new_trust <= 3:
                            # Trust too low — delete the old fact
                            try:
                                semantic.delete(ids=[existing_id])
                                print(f"[Semantic] Fact DELETED — trust dropped to {new_trust}")
                            except:
                                pass
                        else:
                            # Update trust score of existing fact
                            try:
                                semantic.update(
                                    ids=[existing_id],
                                    metadatas=[{
                                        **existing_meta,
                                        "trust_score": new_trust,
                                        "flagged": new_trust <= 5,
                                        "contradicted_count": existing_meta.get('contradicted_count', 0) + 1
                                    }]
                                )
                            except:
                                pass
                    else:
                        # Same fact confirmed — increment trust score
                        new_trust = min(10, existing_trust + 1)
                        new_confirmed = existing_confirmed + 1
                        print(f"[Semantic] Fact CONFIRMED — trust: {existing_trust} → {new_trust}")
                        try:
                            semantic.update(
                                ids=[existing_id],
                                metadatas=[{
                                    **existing_meta,
                                    "trust_score": new_trust,
                                    "confirmed_count": new_confirmed
                                }]
                            )
                        except:
                            pass
                        return  # Don't save duplicate

    # Step 2 — save new fact with full provenance
    semantic.upsert(
        documents=[fact],
        embeddings=[embedding],
        ids=[fact_id],
        metadatas=[{
            "fact_id": fact_id,
            "confidence": confidence,
            "trust_score": confidence,       # starts at initial confidence
            "confirmed_count": 1,            # how many debates confirmed this
            "contradicted_count": 0,         # how many debates contradicted this
            "flagged": False,                # flagged if trust drops below 5
            "source_question": source_question,
            "debate_id": debate_id,
            "timestamp": timestamp
        }]
    )
    print(f"[Semantic] New fact saved — trust: {confidence} | {fact[:60]}...")

def retrieve_semantic(question: str) -> list:
    """
    Retrieves semantic facts.
    Only returns facts that are not flagged and have trust score above 4.
    """
    if semantic.count() == 0:
        return []

    query_embedding = embedder.encode(question).tolist()
    results = semantic.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K_MEMORIES, semantic.count()),
        include=["documents", "distances", "metadatas"]
    )

    facts = []
    if results['documents']:
        for doc, distance, meta in zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        ):
            trust = meta.get('trust_score', 5.0)
            flagged = meta.get('flagged', False)

            # Only return trusted, unflagged facts
            if distance < 1.0 and trust > 4 and not flagged:
                facts.append(f"{doc} [trust:{trust:.1f}]")
                print(f"[Semantic] Retrieved fact — trust: {trust:.1f} | distance: {distance:.3f}")
            elif flagged:
                print(f"[Semantic] SKIPPED flagged fact — trust: {trust:.1f}")
            elif trust <= 4:
                print(f"[Semantic] SKIPPED low trust fact — trust: {trust:.1f}")

    return facts

# ----------------------------------------------------------------
# PROCEDURAL MEMORY — evidence gated
# Only saves when confidence high AND evidence is specific
# ----------------------------------------------------------------

def is_evidence_specific(evidence: str) -> bool:
    """
    Checks if evidence is specific and factual.
    Rejects vague assertive language.
    """
    if not evidence or len(evidence.split()) < 5:
        return False

    vague_phrases = [
        "obviously", "clearly", "everyone knows", "it is well known",
        "undoubtedly", "certainly", "absolutely", "definitely",
        "this is true", "this is false", "common knowledge",
        "it is obvious", "as we all know", "without doubt",
        "it is clear", "surely", "of course", "naturally"
    ]

    evidence_lower = evidence.lower()
    for phrase in vague_phrases:
        if phrase in evidence_lower:
            return False

    # Specific evidence has numbers, proper nouns, or dates
    has_numbers = bool(re.search(r'\d+', evidence))
    has_proper_nouns = evidence != evidence.lower()
    is_long_enough = len(evidence.split()) >= 5

    return (has_numbers or has_proper_nouns) and is_long_enough

def save_procedural(strategy: str, winning_side: str, question: str,
                    confidence: float, evidence: str, debate_id: str):
    """
    Saves winning strategy ONLY if:
    1. Judge confidence >= 8
    2. Winning argument had specific evidence
    """
    if confidence < 8:
        print(f"[Procedural] SKIPPED — confidence {confidence} too low")
        return False

    if not is_evidence_specific(evidence):
        print(f"[Procedural] SKIPPED — evidence not specific enough: {evidence[:60]}")
        return False

    content = f"Strategy: {strategy} | Winner: {winning_side} | Evidence: {evidence} | Question: {question}"
    embedding = embedder.encode(content).tolist()
    strategy_id = (strategy + debate_id).replace(" ", "_")[:50]

    procedural.upsert(
        documents=[content],
        embeddings=[embedding],
        ids=[strategy_id],
        metadatas=[{
            "winning_side": winning_side,
            "question": question,
            "confidence": confidence,
            "evidence": evidence,
            "debate_id": debate_id,
            "timestamp": datetime.now().isoformat()
        }]
    )
    print(f"[Procedural] Strategy saved — evidence verified specific")
    return True

def retrieve_procedural(question: str) -> list:
    if procedural.count() == 0:
        return []

    query_embedding = embedder.encode(question).tolist()
    results = procedural.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K_MEMORIES, procedural.count()),
        include=["documents", "distances"]
    )

    strategies = []
    if results['documents']:
        for doc, distance in zip(results['documents'][0], results['distances'][0]):
            if distance < 1.0:
                strategies.append(doc)
    return strategies

# ----------------------------------------------------------------
# MASTER RETRIEVE — all 4 tiers
# ----------------------------------------------------------------

def retrieve_all_memories(question: str, debate_id: str) -> dict:
    print(f"\n[Memory] Retrieving from all 4 tiers...")
    memories = {
        "short_term":  get_short_term(debate_id),
        "episodic":    retrieve_episodic(question),
        "semantic":    retrieve_semantic(question),
        "procedural":  retrieve_procedural(question)
    }
    total = sum(len(v) for v in memories.values())
    print(f"[Memory] Total retrieved: {total} items")
    return memories

# ----------------------------------------------------------------
# MASTER SAVE — all tiers with all checks
# ----------------------------------------------------------------

def save_all_memories(question: str, transcript: str,
                      verdict_structured: dict, debate_id: str,
                      threshold: float = 7.0):

    confidence    = verdict_structured.get("confidence", 0)
    final_answer  = verdict_structured.get("final_answer", "")
    winning_side  = verdict_structured.get("winning_side", "")
    key_fact      = verdict_structured.get("key_fact", "")
    strategy      = verdict_structured.get("winning_strategy", "")
    evidence      = verdict_structured.get("winning_evidence", "")

    print(f"\n[Memory] Saving — confidence: {confidence}")

    # Episodic — threshold gated
    save_episodic(question, transcript, final_answer, confidence, threshold)

    # Semantic — with trust scoring and contradiction detection
    if key_fact and confidence >= threshold:
        save_semantic(key_fact, confidence, question, debate_id)

    # Procedural — evidence gated
    if strategy and winning_side and evidence:
        save_procedural(strategy, winning_side, question,
                       confidence, evidence, debate_id)

    # Clear short term
    clear_short_term(debate_id)

    print(f"[Memory] All tiers updated.")

# ----------------------------------------------------------------
# GROUND TRUTH CORRECTION
# Call this when Judge answer is wrong
# Stores correct answer directly into semantic memory with trust 10
# ----------------------------------------------------------------

def save_ground_truth_correction(question: str, correct_answer: str, debate_id: str):
    """
    When Judge is wrong, store correct answer as ground truth fact.
    Trust score 10 — highest possible, cannot be overwritten by agent answers.
    """
    fact = f"GROUND TRUTH for '{question}': {correct_answer}"
    print(f"[Memory] Saving ground truth correction: {fact[:80]}...")
    save_semantic(fact, confidence=10.0, source_question=question, debate_id=debate_id)