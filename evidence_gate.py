
import math
import re
from typing import List, Dict
import json
import os
from datetime import datetime

# ----------------------------------------------------------------
# SIGNAL 1: Shannon Entropy — Agent Agreement
# ----------------------------------------------------------------

def compute_entropy(claims: List[Dict]) -> float:
    """
    Measures how much agents agree on a factual direction.
    Low entropy = strong agreement = more trustworthy signal.
    High entropy = agents split = uncertain claim.
    
    claims: list of structured agent messages from one debate round
    Each claim has a 'contradicts' field — if non-empty, that agent opposes
    """
    total = len(claims)
    if total == 0:
        return 1.0  # maximum uncertainty if no claims

    # Count agents who SUPPORT vs OPPOSE the majority claim
    # Simple proxy: agents whose 'contradicts' is empty = supporting
    # Agents whose 'contradicts' is non-empty and specific = opposing
    supporting = sum(
        1 for c in claims
        if not c.get('contradicts') or c.get('contradicts', '').lower() in ['none', 'n/a', '']
    )
    opposing = total - supporting

    p_support = supporting / total
    p_oppose = opposing / total

    # Shannon entropy formula: H = -sum(p * log2(p))
    H = 0.0
    for p in [p_support, p_oppose]:
        if p > 0:
            H -= p * math.log2(p)

    return H  # range: 0.0 (perfect agreement) to 1.0 (perfectly split)

# ----------------------------------------------------------------
# SIGNAL 2: Specificity Score
# ----------------------------------------------------------------

HEDGE_WORDS = [
    "might", "possibly", "perhaps", "i think", "probably",
    "seems", "appears", "allegedly", "potentially", "could be",
    "may", "speculated", "rumored", "unclear", "uncertain",
    "supposedly", "reportedly", "believed to"
]

def compute_specificity(claim: str, evidence: str) -> float:
    """
    Measures how specific and verifiable a claim+evidence pair is.
    Three components:
    1. Lexical specificity: numbers, dates, proper nouns present
    2. Hedging penalty: vague speculative language present
    3. Length bonus: very short evidence is usually vague

    Returns score 0.0 to 1.0
    """
    text = (claim + " " + evidence).lower()
    full_text = claim + " " + evidence  # keep original case for proper noun check

    # Component 1: Lexical specificity
    # Check for numbers (including decimals, percentages)
    has_numbers = bool(re.search(r'\b\d+\.?\d*\b', text))

    # Check for dates (years, months)
    has_dates = bool(re.search(
        r'\b(19|20)\d{2}\b|\b(january|february|march|april|may|june|'
        r'july|august|september|october|november|december)\b',
        text
    ))

    # Check for proper nouns (capitalized words not at start of sentence)
    words = full_text.split()
    proper_nouns = [
        w for i, w in enumerate(words)
        if i > 0 and w[0].isupper() and len(w) > 2 and w.lower() not in
        ['the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or']
    ]
    has_proper_nouns = len(proper_nouns) > 0

    # Lexical score: at least one of numbers/dates/proper nouns
    lex_score = 0.0
    if has_numbers:
        lex_score += 0.4
    if has_dates:
        lex_score += 0.3
    if has_proper_nouns:
        lex_score += 0.3
    lex_score = min(1.0, lex_score)

    # Component 2: Hedging penalty
    hedge_count = sum(1 for h in HEDGE_WORDS if h in text)
    hedge_penalty = min(1.0, hedge_count * 0.2)  # each hedge word costs 0.2, max penalty 1.0
    hedging_score = 1.0 - hedge_penalty

    # Component 3: Length bonus (very short evidence is usually vague)
    evidence_words = len(evidence.split())
    if evidence_words < 3:
        length_score = 0.3   # too short, probably vague
    elif evidence_words < 6:
        length_score = 0.7   # acceptable
    else:
        length_score = 1.0   # good length

    # Combine: 40% lexical, 40% hedging, 20% length
    specificity = 0.40 * lex_score + 0.40 * hedging_score + 0.20 * length_score

    return round(specificity, 3)

# ----------------------------------------------------------------
# GATE DECISION
# ----------------------------------------------------------------

def evidence_gate(
    claims: List[Dict],
    entropy_threshold: float = 0.95,
    specificity_threshold: float = 0.50
) -> List[Dict]:
    """
    Filters agent claims. Returns only claims that pass the gate.

    entropy_threshold: reject if entropy ABOVE this (too much disagreement)
    specificity_threshold: reject if specificity BELOW this (too vague)

    Note: entropy is computed across ALL claims together.
    Specificity is computed per individual claim.
    """
    if not claims:
        return []

    # Compute entropy across all claims in this round
    H = compute_entropy(claims)

    passed = []
    rejected = []

    for claim in claims:
        claim_text = claim.get('claim', '')
        evidence_text = claim.get('evidence', '')

        # Compute specificity for this individual claim
        spec = compute_specificity(claim_text, evidence_text)

        # Gate decision
        passes_entropy = H <= entropy_threshold
        passes_specificity = spec >= specificity_threshold

        gate_result = {
            **claim,
            'gate_entropy': round(H, 3),
            'gate_specificity': spec,
            'gate_passed': passes_entropy and passes_specificity,
            'gate_reason': (
                'passed' if (passes_entropy and passes_specificity)
                else f"failed: {'entropy=' + str(round(H,3)) + ' too high ' if not passes_entropy else ''}"
                     f"{'specificity=' + str(spec) + ' too low' if not passes_specificity else ''}"
            )
        }
        claim['gate_result'] = gate_result

        if passes_entropy and passes_specificity:
            passed.append(claim)
        else:
            rejected.append(claim)
        

    print(f"\n[Evidence Gate] Entropy: {H:.3f} | "
          f"Passed: {len(passed)}/{len(claims)} claims")
    for r in rejected:
        print(f"  REJECTED: {r.get('claim','')[:60]} | "
              f"Reason: {r['gate_result']['gate_reason']}")
    for p in passed:
        print(f"  PASSED:   {p.get('claim','')[:60]} | "
              f"Specificity: {p['gate_result']['gate_specificity']:.3f}")

    return passed

# ----------------------------------------------------------------
# EVALUATION: Precision/Recall on gate decisions
# ----------------------------------------------------------------

def evaluate_gate(
    test_cases: List[Dict],
    entropy_threshold: float = 0.95,
    specificity_threshold: float = 0.50
) -> Dict:
    """
    Evaluates gate performance on labeled test cases.

    test_cases: list of dicts with keys:
        - 'claims': list of structured agent messages
        - 'ground_truth_label': 1 if claims contain correct facts, 0 if wrong/vague
    
    Returns precision, recall, F1 on storage decisions.
    """
    tp = fp = fn = tn = 0

    for case in test_cases:
        if 'claims' in case:
            claims = case['claims']
        else:
            claims = [case]  # wrap single claim in a list
        true_label = case['ground_truth_label']

        passed = evidence_gate(claims, entropy_threshold, specificity_threshold)
        predicted_store = 1 if len(passed) > 0 else 0

        if predicted_store == 1 and true_label == 1:
            tp += 1  # correctly stored good fact
        elif predicted_store == 1 and true_label == 0:
            fp += 1  # wrongly stored bad fact
        elif predicted_store == 0 and true_label == 1:
            fn += 1  # wrongly discarded good fact
        else:
            tn += 1  # correctly discarded bad fact

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n[Gate Evaluation]")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1
    }

# ----------------------------------------------------------------
# BASELINE COMPARISON
# ----------------------------------------------------------------

def baseline_gate(claims: List[Dict]) -> List[Dict]:
    """
    Baseline: fixed threshold on confidence only (your current system).
    Accept any claim with confidence >= 7.
    This is what you compare your evidence gate against.
    """
    passed = [c for c in claims if c.get('confidence', 0) >= 7]
    print(f"[Baseline Gate] Passed: {len(passed)}/{len(claims)} claims "
          f"(confidence >= 7)")
    return passed


def log_gate_decision(claim: Dict, H: float, spec: float, passed: bool):
    """Logs every gate decision with all sub-scores for debugging."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": claim.get('agent', ''),
        "claim": claim.get('claim', '')[:100],
        "confidence": claim.get('confidence', 0),
        "entropy": H,
        "specificity": spec,
        "gate_passed": passed
    }
    with open('gate_log.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def find_optimal_thresholds(labeled_cases: List[Dict]) -> Dict:
    """
    Grid search over entropy and specificity thresholds.
    Finds the combination that maximizes F1 on labeled data.
    No fixed thresholds — learned from your own debate history.
    """
    entropy_candidates = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    specificity_candidates = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70]

    best_f1 = 0.0
    best_thresholds = {"entropy": 0.95, "specificity": 0.50}
    results = []

    for e_thresh in entropy_candidates:
        for s_thresh in specificity_candidates:
            metrics = evaluate_gate(
                labeled_cases,
                entropy_threshold=e_thresh,
                specificity_threshold=s_thresh
            )
            f1 = metrics['f1']
            results.append({
                "entropy_threshold": e_thresh,
                "specificity_threshold": s_thresh,
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "f1": f1
            })
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = {
                    "entropy": e_thresh,
                    "specificity": s_thresh
                }

    print(f"\n[Threshold Search] Best F1: {best_f1:.3f}")
    print(f"[Threshold Search] Optimal thresholds: {best_thresholds}")

    # Save results for paper
    with open('threshold_search_results.json', 'w') as f:
        json.dump({"best": best_thresholds, "all_results": results}, f, indent=2)

    return best_thresholds

if __name__ == "__main__":
    # Quick self-test when run directly
    test_claims = [
        {
            "agent": "Proposer",
            "claim": "Apollo 11 moon landing occurred July 20 1969",
            "reasoning": "NASA mission independently verified worldwide",
            "evidence": "400kg lunar rock samples verified by scientists in 6 countries",
            "confidence": 9,
            "contradicts": "none"
        },
        {
            "agent": "Challenger",
            "claim": "The footage was potentially fabricated",
            "reasoning": "Lighting seems inconsistent with lunar conditions",
            "evidence": "Shadows might be wrong",
            "confidence": 4,
            "contradicts": "Proposer claim: Apollo 11 landing occurred"
        },
        {
            "agent": "Devil's Advocate",
            "claim": "Shadow argument ignores NASA TR-R-200 lens distortion data",
            "reasoning": "Wide-angle lenses produce parallel shadows at lunar distances",
            "evidence": "NASA technical report TR-R-200 documents expected shadow geometry",
            "confidence": 7,
            "contradicts": "Challenger claim: lighting inconsistent"
        }
    ]

    print("=== EVIDENCE GATE ===")
    passed = evidence_gate(test_claims)
    print(f"\nPassed: {len(passed)}/3 claims")

    print("\n=== BASELINE ===")
    baseline_passed = baseline_gate(test_claims)
    print(f"Passed: {len(baseline_passed)}/3 claims")
    print("\n=== THRESHOLD SEARCH ===")
    # Load labeled data
    import json as _json
    with open('labeled_claims.json') as f:
        labeled_cases = _json.load(f)
    
    print(f"Loaded {len(labeled_cases)} labeled claims")
    optimal = find_optimal_thresholds(labeled_cases)
    print(f"\nOptimal thresholds found: {optimal}")
