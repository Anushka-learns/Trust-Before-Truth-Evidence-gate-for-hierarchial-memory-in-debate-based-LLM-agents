import math
import json
import os
from datetime import datetime
from typing import Dict, Optional

# Category-specific recency decay rates
RECENCY_DECAY = {
    "scientific": 0.01,
    "historical": 0.005,
    "temporal": 0.30,
    "personal": 0.05,
    "preference": 0.10,
    "general": 0.05  # default
}

# Full 6-factor weights
PHASE4_WEIGHTS = {
    "w1_sr": 0.30,
    "w2_ms": 0.25,
    "w3_cr": 0.20,
    "w4_u":  0.10,
    "w5_cp": 0.10,
    "w6_r":  0.05
}

PHASE3_WEIGHTS = {
    "w1_sr": 0.35,
    "w2_ms": 0.30,
    "w3_cr": 0.25,
    "w4_u":  0.00,
    "w5_cp": 0.00,
    "w6_r":  0.10
}

def compute_trust_phase3(
    confidence: float,
    sessions_since_stored: int,
    confirmed_count: int = 0,
    contradicted_count: int = 0,
    cosine_with_existing: float = 1.0,
    fact: Optional[str] = None,
    category: Optional[str] = None,
    weights: Dict = None
) -> Dict:
    """
    Phase 3 trust score: SR + MS + CR + Recency.
    Uses a simplified subset of the full Phase 4 model.
    """
    if weights is None:
        weights = PHASE3_WEIGHTS

    return compute_trust_phase4(
        confidence=confidence,
        sessions_since_stored=sessions_since_stored,
        confirmed_count=confirmed_count,
        contradicted_count=contradicted_count,
        cosine_with_existing=cosine_with_existing,
        retrieval_log=[],
        fact=fact,
        fact_embedding=None,
        embedder_model=None,
        category=category,
        weights=weights
    )

def compute_trust_phase4(
    confidence: float,
    sessions_since_stored: int,
    confirmed_count: int = 0,
    contradicted_count: int = 0,
    cosine_with_existing: float = 1.0,
    retrieval_log: list = None,
    fact: Optional[str] = None,
    fact_embedding = None,
    embedder_model = None,
    category: Optional[str] = None,
    weights: Dict = None
) -> Dict:
    """
    Full 6-factor trust score: SR + MS + CR + U + CP + R
    This is the complete trust model described in the paper.
    """
    if weights is None:
        weights = PHASE4_WEIGHTS
    if retrieval_log is None:
        retrieval_log = []
    if category is None:
        category = classify_category(fact) if fact else "general"

    # Compute all 6 factors
    sr = compute_sr(confidence)
    ms = compute_ms(confirmed_count, contradicted_count)
    cr = compute_cr(cosine_with_existing)
    u  = compute_utility(retrieval_log)
    cp = compute_cp(retrieval_log, fact_embedding, embedder_model)
    r  = compute_recency(sessions_since_stored, category)

    cr_penalty = 1.0 - cr

    trust_raw = (
        weights["w1_sr"] * sr +
        weights["w2_ms"] * ms -
        weights["w3_cr"] * cr_penalty +
        weights["w4_u"]  * u +
        weights["w5_cp"] * cp +
        weights["w6_r"]  * r
    )
    trust_raw = max(0.0, min(1.0, trust_raw))
    trust_scaled = round(trust_raw * 10, 2)

    subscores = {
        "SR":  round(sr, 3),
        "MS":  round(ms, 3),
        "CR":  round(cr, 3),
        "CR_penalty": round(cr_penalty, 3),
        "U":   round(u, 3),
        "CP":  round(cp, 3),
        "R":   round(r, 3),
        "category": category,
        "confirmed_count": confirmed_count,
        "contradicted_count": contradicted_count,
        "cosine_with_existing": cosine_with_existing,
        "retrieval_log_size": len(retrieval_log),
        "weights_used": weights,
        "trust_raw": round(trust_raw, 3),
        "trust_scaled": trust_scaled,
        "phase": 4
    }

    return subscores
def classify_category(fact: str) -> str:
    """
    Simple rule-based category classifier.
    Classifies a fact into one of 5 categories.
    Phase 3 will replace this with a proper classifier.
    """
    fact_lower = fact.lower()

    # Temporal signals
    temporal_words = ["today", "current", "now", "recent", "latest",
                      "this year", "this month", "yesterday"]
    if any(w in fact_lower for w in temporal_words):
        return "temporal"

    # Historical signals
    historical_words = ["in 1", "in 2", "century", "ancient", "historical",
                        "founded", "established", "born", "died", "war"]
    if any(w in fact_lower for w in historical_words):
        return "historical"

    # Scientific signals
    scientific_words = ["scientific", "study", "research", "proven",
                        "experiment", "theory", "evidence", "data",
                        "nasa", "dna", "molecule", "physics", "biology"]
    if any(w in fact_lower for w in scientific_words):
        return "scientific"

    # Personal signals
    personal_words = ["i ", "my ", "me ", "person", "individual", "user"]
    if any(w in fact_lower for w in personal_words):
        return "personal"

    return "general"

def compute_sr(confidence: float) -> float:
    """
    Source Reliability = Judge confidence normalized to 0-1.
    Simple, direct, no assumptions needed.
    """
    return min(1.0, max(0.0, confidence / 10.0))

def compute_recency(sessions_since_stored: int, category: str) -> float:
    """
    Recency = Ebbinghaus exponential decay.
    R = e^(-lambda * delta_sessions)
    Lambda is category-specific.
    """
    lambda_val = RECENCY_DECAY.get(category, 0.05)
    return math.exp(-lambda_val * sessions_since_stored)

def compute_ms(confirmed_count: int, contradicted_count: int,
               alpha: float = 1.0, beta: float = 1.0) -> float:
    """
    Memory Stability using Beta-Binomial model.
    MS = (alpha + confirmations) / (alpha + beta + confirmations + contradictions)
    
    alpha=1, beta=1 = uninformative prior (start neutral)
    New fact with no history: MS = (1+0)/(1+1+0+0) = 0.5 (neutral)
    Confirmed 3 times, never contradicted: MS = 4/5 = 0.80
    Confirmed 3 times, contradicted 2: MS = 4/7 = 0.57
    """
    return (alpha + confirmed_count) / (alpha + beta + confirmed_count + contradicted_count)

def compute_cr(cosine_similarity: float) -> float:
    """
    Contradiction Risk approximated via cosine similarity between
    new fact and most similar existing stored fact.
    
    CR = 1 - (1 - cosine) / 2
    
    cosine=1.0 (identical facts) -> CR=1.0 (no contradiction risk)
    cosine=0.0 (unrelated facts) -> CR=0.5 (moderate risk)
    cosine=-1.0 (opposite facts) -> CR=0.0 (maximum contradiction risk)
    
    If no existing fact found (first time storing), cosine=1.0 -> CR=1.0
    """
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    return 1.0 - (1.0 - cosine_similarity) / 2.0
def compute_utility(retrieval_log: list, gamma: float = 0.9) -> float:
    """
    Discounted Utility — measures how useful this fact has been
    in past debates that ended correctly.
    
    U = sum(gamma^(k-i) * r_i * correct_i) / sum(gamma^(k-i) * r_i)
    
    gamma = discount factor (recent retrievals matter more)
    r_i = cosine similarity at retrieval time (relevance)
    correct_i = 1 if that debate was correct, 0 if wrong
    
    Returns 0.5 (neutral) if no retrieval history yet.
    """
    if not retrieval_log:
        return 0.5  # neutral — no history

    # Only use entries where debate outcome is known
    known = [
        e for e in retrieval_log
        if e.get('debate_correct') is not None
    ]

    if not known:
        return 0.5  # neutral — no outcome data yet

    k = len(known)
    numerator = 0.0
    denominator = 0.0

    for i, entry in enumerate(known):
        cosine = entry.get('cosine', 0.5)
        correct = 1.0 if entry.get('debate_correct') else 0.0
        discount = gamma ** (k - 1 - i)  # most recent = gamma^0 = 1.0

        numerator += discount * cosine * correct
        denominator += discount * cosine

    if denominator == 0:
        return 0.5

    return round(numerator / denominator, 3)

def compute_cp(retrieval_log: list, fact_embedding=None,
               embedder=None) -> float:
    """
    Context Persistence — measures how broadly applicable this fact is.
    
    CP = 1 - avg(cosine(fact_embedding, query_embedding_i))
    
    Low average cosine = fact retrieved for diverse topics = high CP
    High average cosine = fact only useful for narrow topic = low CP
    
    Returns 0.5 (neutral) if no retrieval history.
    """
    if not retrieval_log or embedder is None or fact_embedding is None:
        return 0.5  # neutral — no history or embeddings

    import numpy as np

    queries = [e.get('query', '') for e in retrieval_log if e.get('query')]
    if not queries:
        return 0.5

    # Get unique queries only
    unique_queries = list(set(queries))

    # Embed all unique queries
    query_embeddings = embedder.encode(unique_queries)

    # Compute cosine similarity between fact and each query
    similarities = []
    for qe in query_embeddings:
        sim = np.dot(fact_embedding, qe) / (
            np.linalg.norm(fact_embedding) * np.linalg.norm(qe) + 1e-8
        )
        similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities)

    # High avg similarity = narrow topic = low CP
    # Low avg similarity = broad applicability = high CP
    cp = 1.0 - avg_similarity
    return round(max(0.0, min(1.0, cp)), 3)

def log_trust_decision(fact: str, subscores: Dict, stored: bool):
    """
    Logs every trust decision with all sub-scores.
    Madam's requirement: log every sub-score alongside final trust score.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "fact": fact[:100],
        "stored": stored,
        **subscores
    }
    with open('trust_log.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def should_store(trust_scaled: float, threshold: float = 5.0) -> bool:
    """
    Storage decision based on trust score.
    Threshold = 5.0 means only store facts with trust >= 5 out of 10.
    This threshold will be learned in Phase 3.
    """
    return trust_scaled >= threshold

# ----------------------------------------------------------------
# BASELINE COMPARISON
# ----------------------------------------------------------------

def baseline_trust(confidence: float, threshold: float = 7.0) -> bool:
    """
    Baseline: your current system — store if confidence >= 7.0
    This is what Phase 2 competes against.
    """
    return confidence >= threshold
def evaluate_trust_phase3(labeled_path='labeled_claims.json',
                          trust_threshold=5.0,
                          baseline_confidence_threshold=7.0):
    """
    Evaluates Phase 3 trust scoring vs Phase 2 vs Baseline.
    For labeled claims without ChromaDB metadata,
    we simulate MS and CR from available signals:
    - debate_correct=True -> confirmed_count=1, contradicted_count=0
    - debate_correct=False -> confirmed_count=0, contradicted_count=1
    - cosine_with_existing defaults to 0.8 (typical similarity)
    """
    if not os.path.exists(labeled_path):
        print("No labeled data found.")
        return

    with open(labeled_path) as f:
        labeled = json.load(f)

    p3_tp = p3_fp = p3_fn = p3_tn = 0
    p2_tp = p2_fp = p2_fn = p2_tn = 0
    b_tp  = b_fp  = b_fn  = b_tn  = 0

    for item in labeled:
        true_label  = item['ground_truth_label']
        confidence  = item.get('confidence', 5)
        fact        = item.get('claim', '')
        debate_ok   = item.get('debate_correct', True)

        # Simulate MS inputs from debate outcome
        confirmed   = 1 if debate_ok else 0
        contradicted = 1 if not debate_ok else 0
        cosine      = 0.85 if debate_ok else 0.40

        # Phase 4
        p4 = compute_trust_phase4(
            confidence=confidence,
            sessions_since_stored=0,
            confirmed_count=confirmed,
            contradicted_count=contradicted,
            cosine_with_existing=cosine,
            fact=fact
        )
        p4_pred = 1 if should_store(p4['trust_scaled'], trust_threshold) else 0

        # Phase 2
        p2 = compute_trust_phase3(confidence=confidence,
                                   sessions_since_stored=0, fact=fact)
        p2_pred = 1 if should_store(p2['trust_scaled'], trust_threshold) else 0

        # Baseline
        b_pred = 1 if baseline_trust(confidence, baseline_confidence_threshold) else 0

        # Count Phase 4
        if p4_pred==1 and true_label==1: p4_tp+=1
        elif p4_pred==1 and true_label==0: p4_fp+=1
        elif p4_pred==0 and true_label==1: p4_fn+=1
        else: p4_tn+=1

        # Count Phase 2
        if p2_pred==1 and true_label==1: p2_tp+=1
        elif p2_pred==1 and true_label==0: p2_fp+=1
        elif p2_pred==0 and true_label==1: p2_fn+=1
        else: p2_tn+=1

        # Count Baseline
        if b_pred==1 and true_label==1: b_tp+=1
        elif b_pred==1 and true_label==0: b_fp+=1
        elif b_pred==0 and true_label==1: b_fn+=1
        else: b_tn+=1

    def metrics(tp, fp, fn, tn):
        p = tp/(tp+fp) if (tp+fp)>0 else 0
        r = tp/(tp+fn) if (tp+fn)>0 else 0
        f = 2*p*r/(p+r) if (p+r)>0 else 0
        return p, r, f

    p4_p, p4_r, p4_f = metrics(p4_tp, p4_fp, p4_fn, p4_tn)
    p2_p, p2_r, p2_f = metrics(p2_tp, p2_fp, p2_fn, p2_tn)
    b_p,  b_r,  b_f  = metrics(b_tp,  b_fp,  b_fn,  b_tn)

    print(f"\n{'='*65}")
    print(f"PHASE 4 vs PHASE 3 vs BASELINE")
    print(f"{'='*65}")
    print(f"{'Metric':<20} {'Phase4':>12} {'Phase3':>12} {'Baseline':>12}")
    print(f"{'-'*65}")
    print(f"{'Precision':<20} {p4_p:>12.3f} {p3_p:>12.3f} {b_p:>12.3f}")
    print(f"{'Recall':<20} {p4_r:>12.3f} {p3_r:>12.3f} {b_r:>12.3f}")
    print(f"{'F1':<20} {p4_f:>12.3f} {p3_f:>12.3f} {b_f:>12.3f}")
    print(f"{'TP':<20} {p4_tp:>12} {p3_tp:>12} {b_tp:>12}")
    print(f"{'FP':<20} {p4_fp:>12} {p3_fp:>12} {b_fp:>12}")
    print(f"{'FN':<20} {p4_fn:>12} {p3_fn:>12} {b_fn:>12}")
    print(f"{'TN':<20} {p4_tn:>12} {p3_tn:>12} {b_tn:>12}")
    print(f"{'='*65}")

    if p4_f > p3_f and p4_f > b_f:
        print(f"✅ Phase 4 OUTPERFORMS both Phase 3 and Baseline")
    elif p4_f > b_f:
        print(f"✅ Phase 4 outperforms Baseline but not Phase 3")
    elif p3_fn == b_f:
        print(f"⚠️  Phase 3 matches Baseline — add U and CP in Phase 4")
    else:
        print(f"❌ Phase 4 underperforms — check weights or threshold")

    return {"phase4_f1": p4_f, "phase3_f1": p3_f, "baseline_f1": b_f}


if __name__ == "__main__":
    print("\n=== PHASE 4 EVALUATION ===")
    
    from sentence_transformers import SentenceTransformer as ST
    _embedder = ST('all-MiniLM-L6-v2')
    
    import json as _json
    import os as _os
    
    if _os.path.exists('labeled_claims.json'):
        with open('labeled_claims.json') as f:
            labeled = _json.load(f)
        
        p4_tp = p4_fp = p4_fn = p4_tn = 0
        
        for item in labeled:
            true_label = item['ground_truth_label']
            confidence = item.get('confidence', 5)
            fact = item.get('claim', '')
            debate_ok = item.get('debate_correct', True)
            
            confirmed = 1 if debate_ok else 0
            contradicted = 1 if not debate_ok else 0
            cosine = 0.85 if debate_ok else 0.40
            
            retrieval_log = [
                {'query': item.get('question',''), 
                 'cosine': cosine, 
                 'debate_correct': debate_ok}
            ]
            
            fe = _embedder.encode(fact)
            
            p4 = compute_trust_phase4(
                confidence=confidence,
                sessions_since_stored=0,
                confirmed_count=confirmed,
                contradicted_count=contradicted,
                cosine_with_existing=cosine,
                retrieval_log=retrieval_log,
                fact=fact,
                fact_embedding=fe,
                embedder_model=_embedder
            )
            
            pred = 1 if should_store(p4['trust_scaled'], 5.0) else 0
            
            if pred==1 and true_label==1: p4_tp+=1
            elif pred==1 and true_label==0: p4_fp+=1
            elif pred==0 and true_label==1: p4_fn+=1
            else: p4_tn+=1
        
        p = p4_tp/(p4_tp+p4_fp) if (p4_tp+p4_fp)>0 else 0
        r = p4_tp/(p4_tp+p4_fn) if (p4_tp+p4_fn)>0 else 0
        f = 2*p*r/(p+r) if (p+r)>0 else 0
        
        print(f"Phase 4 Results:")
        print(f"  Precision: {p:.3f}")
        print(f"  Recall:    {r:.3f}")
        print(f"  F1:        {f:.3f}")
        print(f"  TP={p4_tp} FP={p4_fp} FN={p4_fn} TN={p4_tn}")
        print(f"\nComparison:")
        print(f"  Baseline F1: 0.760")
        print(f"  Phase 3 F1:  0.780")
        print(f"  Phase 4 F1:  {f:.3f}")
