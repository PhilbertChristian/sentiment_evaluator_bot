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

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Paths / config (can be overridden via CLI)
TRANSCRIPT = Path("data/video_transcripts.jsonl")
EMB_MODEL = "all-MiniLM-L6-v2"
COLLECTION_FINE = "headon_gpt_fine"      # utterance-level
COLLECTION_COARSE = "headon_gpt_coarse"  # ~300-word segments
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


def build_index(transcript_path: Path, reset: bool = False, segment_words: int = 300):
    """Build dual indexes: fine (utterance) and coarse (~segment_words words)."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(CHROMA_DIR))
    if reset:
        for name in (COLLECTION_FINE, COLLECTION_COARSE):
            try:
                client.delete_collection(name)
                print(f"🔁 Reset collection '{name}'")
            except Exception as e:
                print(f"⚠️ Could not reset collection '{name}': {e}")

    fine = client.get_or_create_collection(COLLECTION_FINE)
    coarse = client.get_or_create_collection(COLLECTION_COARSE)
    model = SentenceTransformer(EMB_MODEL)

    docs_f, ids_f, meta_f = [], [], []
    docs_c, ids_c, meta_c = [], [], []
    count_f = 0

    seg_id = 1
    seg_text, seg_start, seg_end = [], None, None
    seg_speakers = set()

    for rec in load_records(transcript_path):
        text = rec.get("text", "")
        if not text:
            continue

        utter_id = rec.get("utteranceId") or rec.get("turn_id") or rec.get("id")
        start = _normalize_time(rec.get("startTime", rec.get("start")))
        end = _normalize_time(rec.get("endTime", rec.get("end")))
        speaker = rec.get("personId") or rec.get("speaker") or "unknown"

        # Fine buffer
        docs_f.append(text)
        ids_f.append(str(utter_id) if utter_id is not None else str(count_f + 1))
        meta_f.append(
            {
                "personId": speaker,
                "startTime": start,
                "endTime": end,
                "segment_id": seg_id,
            }
        )

        # Coarse accumulation
        seg_text.append(text)
        seg_speakers.add(speaker)
        seg_start = start if seg_start is None else min(seg_start, start)
        seg_end = end if seg_end is None else max(seg_end, end)

        # Flush fine batch
        if len(docs_f) >= 128:
            embs = model.encode(docs_f, convert_to_numpy=True).tolist()
            fine.add(documents=docs_f, embeddings=embs, ids=ids_f, metadatas=meta_f)
            count_f += len(docs_f)
            docs_f, ids_f, meta_f = [], [], []

        # Flush coarse when long enough
        if sum(len(t.split()) for t in seg_text) >= segment_words:
            docs_c.append(" ".join(seg_text))
            ids_c.append(f"seg-{seg_id}")
            meta_c.append(
                {
                    "segment_id": seg_id,
                    "speakers": ", ".join(sorted(seg_speakers)) if seg_speakers else "",
                    "startTime": seg_start,
                    "endTime": seg_end,
                }
            )
            seg_id += 1
            seg_text, seg_speakers, seg_start, seg_end = [], set(), None, None

    # Flush remaining fine
    if docs_f:
        embs = model.encode(docs_f, convert_to_numpy=True).tolist()
        fine.add(documents=docs_f, embeddings=embs, ids=ids_f, metadatas=meta_f)
        count_f += len(docs_f)

    # Flush remaining coarse
    if seg_text:
        docs_c.append(" ".join(seg_text))
        ids_c.append(f"seg-{seg_id}")
        meta_c.append(
            {
                "segment_id": seg_id,
                "speakers": ", ".join(sorted(seg_speakers)) if seg_speakers else "",
                "startTime": seg_start,
                "endTime": seg_end,
            }
        )

    if docs_c:
        embs_c = model.encode(docs_c, convert_to_numpy=True).tolist()
        coarse.add(documents=docs_c, embeddings=embs_c, ids=ids_c, metadatas=meta_c)

    print(f"✅ Fine index: {count_f} utterances")
    print(f"✅ Coarse index: {len(ids_c)} segments")
    print(f"Source: {transcript_path} → {CHROMA_DIR}")


def retrieve_snippets(question: str, k: int, model_name: str = EMB_MODEL, k_coarse: int | None = None, k_fine_per_seg: int = 3):
    """
    Two-stage retrieval:
    1) Search coarse segments to pick segment_ids.
    2) Search fine utterances constrained to those segment_ids.
    3) Combine and rerank by distance; return top-k text snippets with metadata.
    """
    client = PersistentClient(path=str(CHROMA_DIR))
    coll_f = client.get_collection(COLLECTION_FINE)
    coll_c = client.get_collection(COLLECTION_COARSE)
    model = SentenceTransformer(model_name)
    emb = model.encode([question], convert_to_numpy=True).tolist()[0]

    if k_coarse is None:
        k_coarse = max(3, k)  # widen coarse search a bit

    # Stage 1: coarse
    res_c = coll_c.query(query_embeddings=[emb], n_results=k_coarse)
    seg_ids = []
    coarse_hits = []
    for doc, meta, dist in zip(res_c["documents"][0], res_c["metadatas"][0], res_c["distances"][0]):
        sid = meta.get("segment_id")
        if sid:
            seg_ids.append(sid)
            coarse_hits.append((dist, doc, meta))

    # Stage 2: fine per segment
    fine_candidates = []
    for sid in seg_ids:
        res_f = coll_f.query(
            query_embeddings=[emb],
            where={"segment_id": sid},
            n_results=k_fine_per_seg,
        )
        for doc, meta, dist in zip(res_f["documents"][0], res_f["metadatas"][0], res_f["distances"][0]):
            fine_candidates.append((dist, doc, meta))

    # Combine coarse+fine, rerank by distance
    all_hits = fine_candidates or []
    all_hits.sort(key=lambda x: x[0])
    top = all_hits[:k]

    snippets = []
    for _, doc, meta in top:
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
