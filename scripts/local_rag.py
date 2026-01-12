"""
Local RAG helper for HeadOn GPT.
- Builds a Chroma index from the transcript JSONL (per-utterance).
- Answers questions by retrieving top-k snippets and routing them to a local Ollama model.

Quick use:
  python scripts/local_rag.py --build --reset --file data/video_transcripts.jsonl
  python scripts/local_rag.py --ask "What was speaker B's stance on a Palestinian state?"
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Paths / config (can be overridden via CLI)
TRANSCRIPT = Path("data/video_transcripts.jsonl")
EMB_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "headon_gpt"
CHROMA_DIR = Path("data/chroma")
OLLAMA_MODEL = "llama3"


def load_records(path: Path):
    if not path.exists():
        raise SystemExit(f"Transcript not found: {path}")
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_time(val):
    if val is None:
        return 0.0
    try:
        v = float(val)
    except Exception:
        return 0.0
    # Heuristic: values > 1000 are likely milliseconds
    return v / 1000.0 if v > 1000 else v


def build_index(transcript_path: Path, reset: bool = False):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
            print(f"🔁 Reset collection '{COLLECTION}'")
        except Exception as e:
            print(f"⚠️ Could not reset collection: {e}")
    coll = client.get_or_create_collection(COLLECTION)
    model = SentenceTransformer(EMB_MODEL)

    docs = []
    ids = []
    meta = []
    count = 0
    for rec in load_records(transcript_path):
        text = rec.get("text", "")
        if not text:
            continue
        docs.append(text)
        utter_id = rec.get("utteranceId") or rec.get("turn_id") or rec.get("id")
        ids.append(str(utter_id) if utter_id is not None else str(count + 1))
        start = _normalize_time(rec.get("startTime", rec.get("start")))
        end = _normalize_time(rec.get("endTime", rec.get("end")))
        speaker = rec.get("personId") or rec.get("speaker") or "unknown"
        meta.append(
            {
                "personId": speaker,
                "startTime": start,
                "endTime": end,
            }
        )
        if len(docs) >= 128:
            embs = model.encode(docs, convert_to_numpy=True).tolist()
            coll.add(documents=docs, embeddings=embs, ids=ids, metadatas=meta)
            count += len(docs)
            docs, ids, meta = [], [], []
    if docs:
        embs = model.encode(docs, convert_to_numpy=True).tolist()
        coll.add(documents=docs, embeddings=embs, ids=ids, metadatas=meta)
        count += len(docs)

    print(f"✅ Index built with {count} docs from {transcript_path} at {CHROMA_DIR}")


def retrieve_snippets(question: str, k: int, model_name: str = EMB_MODEL):
    client = PersistentClient(path=str(CHROMA_DIR))
    coll = client.get_collection(COLLECTION)
    model = SentenceTransformer(model_name)
    emb = model.encode([question], convert_to_numpy=True).tolist()[0]
    res = coll.query(query_embeddings=[emb], n_results=k)
    snippets = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        speaker = meta.get("personId") or meta.get("speaker") or "unknown"
        snippets.append(
            f"[{speaker} {meta.get('startTime')}-{meta.get('endTime')}s] {doc}"
        )
    return snippets


def build_prompt(question: str, snippets: list[str]) -> str:
    return (
        "You are an analyst. Use only the provided snippets to answer.\n\n"
        "Snippets:\n"
        + "\n".join(snippets)
        + f"\n\nQuestion: {question}\nAnswer concisely:"
    )


def run_ollama(prompt: str, model: str) -> str:
    try:
        out = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise SystemExit("ollama CLI not found. Install Ollama or use --only-retrieve to test retrieval.")
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Ollama failed: {e.stderr.decode().strip()}")
    return out.stdout.decode().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="Build the index")
    ap.add_argument("--reset", action="store_true", help="Drop and rebuild the collection before indexing")
    ap.add_argument("--file", type=Path, default=TRANSCRIPT, help="Transcript JSONL path")
    ap.add_argument("--ask", type=str, help="Ask a question")
    ap.add_argument("--k", type=int, default=6, help="Top-k snippets to retrieve")
    ap.add_argument("--llm-model", default=OLLAMA_MODEL, help="Ollama model name to use for answering")
    ap.add_argument("--only-retrieve", action="store_true", help="Only show retrieved snippets, do not call LLM")
    args = ap.parse_args()

    if args.build:
        build_index(args.file, reset=args.reset)
    if args.ask:
        snippets = retrieve_snippets(args.ask, k=args.k)
        if args.only_retrieve:
            print("Retrieved snippets:\n" + "\n".join(snippets))
            return
        prompt = build_prompt(args.ask, snippets)
        answer = run_ollama(prompt, args.llm_model)
        print(answer)
    if not args.build and not args.ask:
        ap.print_help()


if __name__ == "__main__":
    main()
