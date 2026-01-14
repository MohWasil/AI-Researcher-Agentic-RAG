from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict

class BM25Indexer:
    def __init__(self, docs_tokens: List[List[str]], metas: List[Dict]):
        self.bm25 = BM25Okapi(docs_tokens)
        self.metas = metas

    def search(self, query_tokens: List[str], top_k=50):
        scores = self.bm25.get_scores(query_tokens)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [{"index": int(i), "score": float(scores[i]), "meta": self.metas[i]} for i in idxs]

# Hybrid retriever
class HybridRetriever:
    def __init__(self, faiss_index: FaissIndexer, bm25: BM25Indexer, faiss_weight=0.7, bm25_weight=0.3):
        self.faiss = faiss_index
        self.bm25 = bm25
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight

    def retrieve(self, query_text: str, query_emb: np.ndarray, query_tokens: List[str], top_k=10):
        faiss_res = self.faiss.search(query_emb, top_k=50)[0]  # list of candidate dicts
        bm25_res = self.bm25.search(query_tokens, top_k=50)

        # Build dict index -> (faiss_score, bm25_score, meta)
        combined = {}
        for r in faiss_res:
            idx = r["index"]
            combined[idx] = {"faiss": r["score"], "meta": r["meta"], "bm25": 0.0}
        for r in bm25_res:
            idx = r["index"]
            if idx not in combined:
                combined[idx] = {"faiss": 0.0, "meta": r["meta"], "bm25": r["score"]}
            else:
                combined[idx]["bm25"] = r["score"]

        # Normalize scores
        faiss_vals = np.array([v["faiss"] for v in combined.values()]) if combined else np.array([0.0])
        bm25_vals = np.array([v["bm25"] for v in combined.values()]) if combined else np.array([0.0])
        # avoid zero-division
        if faiss_vals.max() > 0: faiss_vals = (faiss_vals - faiss_vals.min())/(faiss_vals.ptp()+1e-12)
        if bm25_vals.max() > 0: bm25_vals = (bm25_vals - bm25_vals.min())/(bm25_vals.ptp()+1e-12)

        final = []
        for i, (idx, v) in enumerate(combined.items()):
            score = self.faiss_weight * float(faiss_vals[i]) + self.bm25_weight * float(bm25_vals[i])
            final.append({"index": idx, "score": score, "meta": v["meta"]})
        final_sorted = sorted(final, key=lambda x: x["score"], reverse=True)[:top_k]
        return final_sorted
