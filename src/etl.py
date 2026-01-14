from src.etl import extract_text_from_file, chunk_text
import json, os

SRC_DIR = "data/samples"
OUT = "data/chunks.jsonl"
os.makedirs("data", exist_ok=True)

with open(OUT, "w", encoding="utf8") as fh:
    for fn in os.listdir(SRC_DIR):
        path = os.path.join(SRC_DIR, fn)
        text, meta = extract_text_from_file(path)
        chunks = chunk_text(text, meta, chunk_size=600, overlap=120)
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
print("Wrote chunks to", OUT)
