"""
fuzzy_engine.py
================
Soft Computing Project — Fuzzy Logic Sentiment Analysis
Mamdani FIS with dataset (Amazon Product Reviews) integration.
"""

import numpy as np
import re
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ─────────────────────────────────────────────────────────────────────────────
#  LEXICONS
# ─────────────────────────────────────────────────────────────────────────────
POSITIVE_WORDS = {
    "good","great","excellent","amazing","wonderful","fantastic","love","happy",
    "best","awesome","beautiful","nice","perfect","brilliant","outstanding",
    "superb","terrific","pleasant","enjoy","enjoyed","positive","helpful",
    "recommend","glad","delighted","impressive","marvelous","fabulous","splendid",
    "magnificent","joyful","cheerful","pleased","satisfied","thrilled","excited",
    "fun","incredible","genius","smooth","clean","fresh","powerful","elegant",
    "efficient","easy","fast","friendly","kind","charming","light","quality",
    "crisp","clear","bright","solid","durable","affordable","value","worth","quick"
}
NEGATIVE_WORDS = {
    "bad","terrible","awful","horrible","worst","hate","ugly","disgusting",
    "disappointing","poor","boring","annoying","frustrating","useless","waste",
    "broken","slow","difficult","complicated","confusing","painful","dreadful",
    "lousy","inferior","mediocre","failure","failed","problem","issue","wrong",
    "defective","unreliable","dangerous","harmful","toxic","corrupt","rude","harsh",
    "aggressive","angry","sad","miserable","depressed","upset","worried","scared",
    "embarrassed","ashamed","bitter","cheap","fragile","glitchy","laggy","ads"
}
INTENSIFIERS = {
    "very","extremely","absolutely","incredibly","really","so","much","highly",
    "totally","deeply","super","truly","quite","pretty","fairly","especially"
}
NEGATORS = {
    "not","never","no","dont","don't","doesn't","doesnt","hardly","barely",
    "neither","nor","without","lack","lacking"
}


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(text: str) -> dict:
    if not text or not isinstance(text, str):
        return {"pos_score":5.0,"neg_score":0.0,"punct_score":0.0,"word_count":0}

    text_clean = re.sub(r"[^a-zA-Z\s'!?]", " ", text.lower())
    words = text_clean.split()
    total = max(len(words), 1)

    pos_count = neg_count = 0.0
    negation_active = False
    negation_window = 0

    for i, w in enumerate(words):
        if w in NEGATORS:
            negation_active = True
            negation_window = 3
        else:
            negation_window -= 1
            if negation_window <= 0:
                negation_active = False

        boost = 1.6 if (i > 0 and words[i-1] in INTENSIFIERS) else 1.0

        if w in POSITIVE_WORDS:
            if negation_active: neg_count += boost
            else:               pos_count += boost
        elif w in NEGATIVE_WORDS:
            if negation_active: pos_count += boost
            else:               neg_count += boost

    exclamation  = min(text.count("!") / max(total, 1) * 10, 1.0)
    caps_ratio   = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    pos_score    = min((pos_count / total) * 10 * 3.0, 10.0)
    neg_score    = min((neg_count / total) * 10 * 3.0, 10.0)
    punct_score  = min((exclamation + caps_ratio) * 5, 10.0)

    return {
        "pos_score":   round(pos_score,  3),
        "neg_score":   round(neg_score,  3),
        "punct_score": round(punct_score,3),
        "word_count":  total,
        "pos_count":   pos_count,
        "neg_count":   neg_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FUZZY INFERENCE SYSTEM  (Mamdani)
# ─────────────────────────────────────────────────────────────────────────────
def build_fis():
    universe           = np.arange(0, 10.01, 0.1)
    sentiment_universe = np.arange(0, 100.01, 0.5)

    pos_score   = ctrl.Antecedent(universe, "positive_score")
    neg_score   = ctrl.Antecedent(universe, "negative_score")
    punct_score = ctrl.Antecedent(universe, "punctuation_score")
    sentiment   = ctrl.Consequent(sentiment_universe, "sentiment")

    # Input MFs
    pos_score["low"]    = fuzz.trimf(universe, [0,  0,  4])
    pos_score["medium"] = fuzz.trimf(universe, [2,  5,  8])
    pos_score["high"]   = fuzz.trimf(universe, [6, 10, 10])

    neg_score["low"]    = fuzz.trimf(universe, [0,  0,  4])
    neg_score["medium"] = fuzz.trimf(universe, [2,  5,  8])
    neg_score["high"]   = fuzz.trimf(universe, [6, 10, 10])

    punct_score["low"]  = fuzz.trimf(universe, [0,  0,  5])
    punct_score["high"] = fuzz.trimf(universe, [4, 10, 10])

    # Output MFs
    sentiment["very_negative"] = fuzz.trimf(sentiment_universe, [0,   0,  20])
    sentiment["negative"]      = fuzz.trimf(sentiment_universe, [10, 25,  40])
    sentiment["neutral"]       = fuzz.trimf(sentiment_universe, [30, 50,  70])
    sentiment["positive"]      = fuzz.trimf(sentiment_universe, [60, 75,  90])
    sentiment["very_positive"] = fuzz.trimf(sentiment_universe, [80,100, 100])

    rules = [
        ctrl.Rule(pos_score["high"]   & neg_score["low"],                           sentiment["very_positive"]),
        ctrl.Rule(pos_score["high"]   & neg_score["low"]   & punct_score["high"],   sentiment["very_positive"]),
        ctrl.Rule(pos_score["high"]   & neg_score["medium"],                        sentiment["positive"]),
        ctrl.Rule(pos_score["medium"] & neg_score["low"],                           sentiment["positive"]),
        ctrl.Rule(pos_score["medium"] & neg_score["low"]   & punct_score["high"],   sentiment["positive"]),
        ctrl.Rule(pos_score["medium"] & neg_score["medium"],                        sentiment["neutral"]),
        ctrl.Rule(pos_score["low"]    & neg_score["low"],                           sentiment["neutral"]),
        ctrl.Rule(pos_score["low"]    & neg_score["low"]   & punct_score["low"],    sentiment["neutral"]),
        ctrl.Rule(neg_score["high"]   & pos_score["low"],                           sentiment["very_negative"]),
        ctrl.Rule(neg_score["high"]   & pos_score["medium"],                        sentiment["negative"]),
        ctrl.Rule(neg_score["medium"] & pos_score["low"],                           sentiment["negative"]),
        ctrl.Rule(neg_score["high"]   & punct_score["high"],                        sentiment["very_negative"]),
        ctrl.Rule(pos_score["high"]   & neg_score["high"],                          sentiment["neutral"]),
        ctrl.Rule(pos_score["low"]    & neg_score["medium"],                        sentiment["negative"]),
        ctrl.Rule(pos_score["medium"] & neg_score["high"],                          sentiment["negative"]),
    ]

    fis_ctrl = ctrl.ControlSystem(rules)
    fis_sim  = ctrl.ControlSystemSimulation(fis_ctrl)

    mf_data = {
        "universe": universe,
        "sentiment_universe": sentiment_universe,
        "pos_score": pos_score,
        "neg_score": neg_score,
        "punct_score": punct_score,
        "sentiment": sentiment,
    }
    return fis_sim, mf_data


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLETON CACHE
# ─────────────────────────────────────────────────────────────────────────────
_fis_sim = None
_mf_data = None

def get_fis():
    global _fis_sim, _mf_data
    if _fis_sim is None:
        _fis_sim, _mf_data = build_fis()
    return _fis_sim, _mf_data


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ANALYSIS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def analyze_sentiment(text: str) -> dict:
    if not text or not str(text).strip():
        return {"error": "Empty input"}

    features       = extract_features(str(text))
    fis_sim, mf    = get_fis()

    ps = np.clip(features["pos_score"],   0.01, 9.99)
    ns = np.clip(features["neg_score"],   0.01, 9.99)
    pu = np.clip(features["punct_score"], 0.01, 9.99)

    fis_sim.input["positive_score"]    = float(ps)
    fis_sim.input["negative_score"]    = float(ns)
    fis_sim.input["punctuation_score"] = float(pu)

    try:
        fis_sim.compute()
        score = float(fis_sim.output["sentiment"])
    except Exception:
        score = float(np.clip((ps - ns + 5) / 10 * 50 + 25, 0, 100))

    if   score >= 80: label, emoji, color = "Very Positive", "😄", "#22c55e"
    elif score >= 60: label, emoji, color = "Positive",      "🙂", "#86efac"
    elif score >= 40: label, emoji, color = "Neutral",       "😐", "#facc15"
    elif score >= 20: label, emoji, color = "Negative",      "😟", "#f97316"
    else:             label, emoji, color = "Very Negative", "😠", "#ef4444"

    su = mf["sentiment_universe"]
    memberships = {
        "Very Negative": float(fuzz.interp_membership(su, mf["sentiment"]["very_negative"].mf, score)),
        "Negative":      float(fuzz.interp_membership(su, mf["sentiment"]["negative"].mf,      score)),
        "Neutral":       float(fuzz.interp_membership(su, mf["sentiment"]["neutral"].mf,       score)),
        "Positive":      float(fuzz.interp_membership(su, mf["sentiment"]["positive"].mf,      score)),
        "Very Positive": float(fuzz.interp_membership(su, mf["sentiment"]["very_positive"].mf, score)),
    }

    return {
        "text":        text,
        "score":       round(score, 2),
        "label":       label,
        "emoji":       emoji,
        "color":       color,
        "features":    features,
        "memberships": memberships,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  RATING → EXPECTED LABEL  (for dataset evaluation)
# ─────────────────────────────────────────────────────────────────────────────
def rating_to_label(rating: float) -> str:
    """Convert star rating to 3-class label for accuracy evaluation."""
    if   rating >= 4: return "Positive"
    elif rating == 3: return "Neutral"
    else:             return "Negative"

def score_to_3class(score: float) -> str:
    if   score >= 55: return "Positive"
    elif score >= 38: return "Neutral"
    else:             return "Negative"
