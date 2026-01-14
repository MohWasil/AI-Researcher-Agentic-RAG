import os
import json
import numpy as np
import faiss
from typing import List, Dict

# Embedding wrapper (supports SentenceTransformers or OpenAI)
class Embedder:
    def __init__(self, model_name="all-mpnet-base-v2", backend="sbert"):
        self.backend = backend
        if backend == "sbert":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
        else:
            # placeholder for OpenAI embeddings
            self.dim = 1536

    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        if self.backend == "sbert":
            embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embs
        else:
            # call OpenAI here (pseudo)
            raise NotImplementedError("Add OpenAI embed integration if desired.")

# FAISS index manager
class FaissIndexer:
    def __init__(self, dim:int, index_path=None, use_ivf=False, nlist=100):
        self.dim = dim
        self.index_path = index_path
        if use_ivf:
            quant = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(dim)
        self.id_to_meta = {}  # store metadata mapping in memory / persist to disk

    def add(self, embeddings: np.ndarray, metas: List[Dict], ids: List[int]=None):
        # embeddings: (N, dim)
        # normalize for cosine if using inner product
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        # store metadata mapping
        start_id = len(self.id_to_meta)
        for i, meta in enumerate(metas):
            idx = start_id + i
            self.id_to_meta[idx] = meta

    def search(self, query_emb: np.ndarray, top_k=10):
        faiss.normalize_L2(query_emb)
        D, I = self.index.search(query_emb, top_k)
        results = []
        for dist_row, idx_row in zip(D, I):
            row = []
            for d, idx in zip(dist_row, idx_row):
                if idx == -1: continue
                meta = self.id_to_meta.get(int(idx), {})
                row.append({"score": float(d), "meta": meta, "index": int(idx)})
            results.append(row)
        return results

    def save(self):
        if self.index_path:
            faiss.write_index(self.index, self.index_path + ".index")
            with open(self.index_path + ".meta.json", "w", encoding="utf8") as f:
                json.dump(self.id_to_meta, f, ensure_ascii=False, indent=2)
