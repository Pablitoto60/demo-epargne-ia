# reco_rules.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

GOAL_KEYWORDS = {
    "retraite": ["retraite", "pension", "fin de carrière", "fin de carriere"],
    "precaution": ["precaution", "précaution", "imprevu", "imprévu", "sécurité", "securite", "coup dur"],
    "immobilier": ["immobilier", "apport", "maison", "appart", "achat"],
    "actions": ["actions", "bourse", "etf", "pea", "compte titre", "compte-titres", "ct0", "cto"],
    "fiscalite": ["impot", "impôts", "déduire", "deduire", "defiscal", "défiscal"],
    "transmission": ["transmission", "succession", "heritage", "héritage"],
    "liquidite": ["retirer", "retrait", "disponible", "liquide", "liquidite", "liquidité"],
}

def infer_goals_from_text(text: str) -> Set[str]:
    t = (text or "").lower()
    goals = set()
    for goal, kws in GOAL_KEYWORDS.items():
        if any(k in t for k in kws):
            goals.add(goal)
    return goals

def recommend_products(
    client: Dict,
    project_text: str,
    catalog: List[Dict],
    top_k: int = 3
) -> Tuple[Dict, List[Dict], Dict]:
    """
    Retourne (recommended, alternatives, debug_info)
    - recommended: dict produit top 1
    - alternatives: liste dict produits top 2/3
    - debug_info: détails utiles pour onglet Règles
    """

    horizon = client.get("horizon_annees")
    risk = client.get("risque")
    if horizon is None or risk is None:
        return None, [], {"reason": "missing_core_slots", "eligible": [], "scores": []}

    goals = infer_goals_from_text(project_text)
    needs_liquidity = "liquidite" in goals or "precaution" in goals

    eligible = []
    scores_debug = []

    for p in catalog:
        # -------- Hard filters --------
        if horizon < p["horizon_min_years"]:
            scores_debug.append({"id": p["id"], "name": p["name"], "excluded": True, "why": "horizon_min"})
            continue

        if p["horizon_max_years"] is not None and horizon > p["horizon_max_years"]:
            horizon_penalty = True
        else:
            horizon_penalty = False

        if risk not in p["risk_profiles_allowed"]:
            scores_debug.append({"id": p["id"], "name": p["name"], "excluded": True, "why": "risk_not_allowed"})
            continue

        if needs_liquidity and p["liquidity"] == "faible":
            scores_debug.append({"id": p["id"], "name": p["name"], "excluded": True, "why": "liquidity_too_low"})
            continue

        # -------- Scoring (soft rules) --------
        score = 0
        prod_goals = set(p.get("goals", []))
        overlap = goals.intersection(prod_goals)

        # 1) Objectif / use-cases
        score += 4 * len(overlap)

        # 2) Horizon bonus/malus
        if horizon <= (p["horizon_min_years"] + 3):
            score += 2
        if horizon_penalty:
            score -= 2

        if p["id"] == "LIVRET_A" and horizon >= 5:
            score -= 4

        # 3) Liquidité
        if needs_liquidity and p["liquidity"] == "forte":
            score += 2
        if needs_liquidity and p["liquidity"] == "moyenne":
            score += 1

        # 4) Pondération priority
        score = score * float(p.get("priority", 1.0))

        eligible.append((score, p))
        scores_debug.append({
            "id": p["id"],
            "name": p["name"],
            "excluded": False,
            "score": round(score, 2),
            "goals_match": sorted(list(overlap)),
            "needs_liquidity": needs_liquidity,
            "risk": risk,
            "horizon": horizon,
        })

    eligible.sort(key=lambda x: x[0], reverse=True)
    if not eligible:
        return None, [], {"reason": "no_eligible_products", "eligible": [], "scores": scores_debug}

    top = [p for _, p in eligible[:top_k]]
    return top[0], top[1:], {
        "reason": "ok",
        "goals": sorted(list(goals)),
        "needs_liquidity": needs_liquidity,
        "scores": scores_debug,
        "eligible": [x["id"] for x in scores_debug if not x.get("excluded")],
    }