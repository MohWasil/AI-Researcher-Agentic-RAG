from typing import List, Dict
import time

class Maker:
    def __init__(self, retriever, llm_client, prompt_template:str, top_k:int=6):
        self.retriever = retriever
        self.llm = llm_client
        self.prompt_template = prompt_template
        self.top_k = top_k

    def make(self, user_query: str, user_context: Dict={}):
        # 1) get query embedding & tokens (llm_client has embedding function)
        query_emb = self.llm.embed([user_query])  # returns shape (1,dim)
        query_tokens = self.llm.tokenize(user_query)

        # 2) retrieve
        candidates = self.retriever.retrieve(user_query, query_emb, query_tokens, top_k=self.top_k)

        # 3) build prompt: include metadata + compact excerpts
        context_snippets = []
        for c in candidates:
            meta = c["meta"]
            excerpt = meta.get("excerpt") or meta.get("text")[:400]
            context_snippets.append(f"### {meta.get('title','unknown')} ({meta.get('doc_id')})\n{excerpt}\n--")

        prompt = self.prompt_template.format(query=user_query, context="\n\n".join(context_snippets))
        # 4) call LLM (deterministic: temperature 0)
        gen = self.llm.generate(prompt, temperature=0.0, max_tokens=512)
        return {
            "answer": gen["text"],
            "provenance": candidates,
            "raw_llm_resp": gen,
            "prompt": prompt,
            "timestamp": time.time()
        }
