import re
from typing import Dict, List

class Checker:
    def __init__(self, claim_extractor=None, safety_rules=None):
        self.claim_extractor = claim_extractor or self.default_claim_extractor
        self.safety_rules = safety_rules or []

    def default_claim_extractor(self, text:str) -> List[str]:
        # Very simple heuristic: sentences with digits / assertions
        sentences = re.split(r'(?<=[.?!])\s+', text)
        claims = [s for s in sentences if re.search(r'\d', s) or len(s.split())>6]
        return claims

    def verify_claim_against_provenance(self, claim:str, provenance:List[Dict]) -> Dict:
        # simple substring search in excerpts (can be improved with QA model)
        for p in provenance:
            excerpt = p["meta"].get("excerpt", "") or p["meta"].get("text", "")
            if claim.strip()[:40] in excerpt:
                return {"ok": True, "evidence": p}
        return {"ok": False, "evidence": None}

    def run_checks(self, candidate:Dict) -> Dict:
        text = candidate["answer"]
        prov = candidate["provenance"]
        claims = self.claim_extractor(text)
        issues = []
        for c in claims:
            res = self.verify_claim_against_provenance(c, prov)
            if not res["ok"]:
                issues.append({"claim": c, "issue": "unsupported", "evidence": None})
        # policy checks (simple)
        for rule in self.safety_rules:
            if rule in text.lower():
                issues.append({"claim": None, "issue": f"policy_violation:{rule}"})
        approved = len(issues) == 0
        confidence = "high" if approved else "low"
        return {"approved": approved, "issues": issues, "confidence": confidence}
