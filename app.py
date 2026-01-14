"""
HeadOn GPT
- Uses Chroma index at data/chroma (built via scripts/local_rag.py)
- Uses OpenAI GPT-4o for high-quality answers
"""

import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from chromadb import PersistentClient
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from scripts.local_rag import (
    build_index,
    CHROMA_DIR,
    COLLECTION_FINE,
    COLLECTION_COARSE,
)
from scripts.youtube_to_jsonl import process_youtube_to_jsonl

DEFAULT_TRANSCRIPT = Path("data/video_transcripts.jsonl")
DEFAULT_OPENAI_MODEL = "gpt-4o"
MODEL_CHOICES = [
    "gpt-5.2",      # placeholder for newest
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
]
LOG_FILE = Path("data/chat_log.jsonl")
SESSION_DIR = Path("data/sessions")
SESSION_INDEX = Path("data/session_index.json")

# Speaker name mapping (customize per video)
SPEAKER_NAMES = {
    "A": "Adam",
    "B": "Tal",
    "speaker_1": "Adam",
    "speaker_2": "Tal",
}


def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    import re
    patterns = [
        r'(?:v=|\/v\/|youtu\.be\/|\/embed\/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def analyze_speakers(transcript_path: Path) -> dict:
    """Analyze transcript to compute speaker stats and identify them."""
    if not transcript_path.exists():
        return {}
    
    speakers = {}
    sample_quotes = {}
    
    with transcript_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            speaker = rec.get("speaker") or rec.get("personId") or "unknown"
            text = rec.get("text", "")
            start = rec.get("start") or rec.get("startTime") or 0
            end = rec.get("end") or rec.get("endTime") or 0
            
            # Normalize to seconds
            if start > 1000:
                start = start / 1000
            if end > 1000:
                end = end / 1000
            
            duration = max(0, end - start)
            words = len(text.split())
            
            if speaker not in speakers:
                speakers[speaker] = {
                    "turns": 0,
                    "words": 0,
                    "duration": 0,
                    "quotes": [],
                }
            
            speakers[speaker]["turns"] += 1
            speakers[speaker]["words"] += words
            speakers[speaker]["duration"] += duration
            
            # Collect longer quotes for identification
            if len(text) > 50 and len(speakers[speaker]["quotes"]) < 5:
                speakers[speaker]["quotes"].append(text[:200])
    
    return speakers


def identify_speaker_with_llm(speaker_id: str, quotes: list[str], openai_key: str = None) -> dict:
    """Use GPT-4o to identify who a speaker might be based on their quotes."""
    if not quotes:
        return {
            "likely_role": "participant",
            "apparent_stance": "See transcript",
            "speaking_style": "conversational",
        }
    
    api_key = openai_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"likely_role": "participant", "apparent_stance": "Add API key to analyze", "speaking_style": "—"}
    
    sample = "\n".join([f"- \"{q}\"" for q in quotes[:3]])
    prompt = f"""Based on these quotes from Speaker {speaker_id} in a debate/conversation, provide a brief analysis.

Quotes:
{sample}

Respond in exactly this JSON format (no other text):
{{"likely_role": "interviewer/guest/host/debater", "apparent_stance": "brief 5-word stance", "speaking_style": "brief 3-word style"}}"""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an analyst. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150,
        )
        result = response.choices[0].message.content.strip()
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    
    return {"likely_role": "participant", "apparent_stance": "See transcript", "speaking_style": "conversational"}

# ─────────────────────────────────────────────────────────────
# Page config & styling
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeadOn GPT",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        * { font-family: 'Inter', sans-serif; }
        
        .stApp {
            background: linear-gradient(180deg, #1a1a1a 0%, #0d0d0d 100%);
        }
        
        /* Hide default streamlit elements */
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 2rem; max-width: 900px; }
        
        /* Greeting */
        .greeting {
            text-align: center;
            padding: 60px 20px 40px;
        }
        .greeting-icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
        .greeting-text {
            font-size: 42px;
            font-weight: 300;
            color: #e8dcc8;
            letter-spacing: -1px;
        }
        .greeting-sub {
            color: #666;
            font-size: 14px;
            margin-top: 8px;
        }
        
        /* Chat input container */
        .chat-container {
            background: #1e1e1e;
            border: 1px solid #333;
            border-radius: 24px;
            padding: 20px 24px;
            margin: 20px auto;
            max-width: 750px;
        }
        
        /* Message bubbles */
        .msg-user {
            background: linear-gradient(135deg, #2d2d2d 0%, #252525 100%);
            border-radius: 20px 20px 4px 20px;
            padding: 16px 20px;
            margin: 12px 0;
            margin-left: 60px;
            color: #fff;
            border: 1px solid #3a3a3a;
        }
        .msg-bot {
            background: linear-gradient(135deg, #1a2332 0%, #151d2a 100%);
            border-radius: 20px 20px 20px 4px;
            padding: 16px 20px;
            margin: 12px 0;
            margin-right: 60px;
            color: #c8d4e3;
            border: 1px solid #2a3444;
        }
        .msg-label {
            font-size: 11px;
            color: #666;
            margin-bottom: 6px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .msg-content {
            line-height: 1.6;
            font-size: 15px;
        }
        
        /* Evidence snippets */
        .evidence {
            background: #0f1419;
            border-radius: 12px;
            padding: 12px 16px;
            margin-top: 12px;
            border-left: 3px solid #d97706;
            font-size: 13px;
            color: #9ca3af;
        }
        .evidence-title {
            color: #d97706;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        
        /* Settings panel */
        .settings-panel {
            background: #151515;
            border-radius: 16px;
            padding: 20px;
            border: 1px solid #252525;
            margin-bottom: 20px;
        }
        .settings-title {
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }
        
        /* K-value badge */
        .k-badge {
            display: inline-block;
            background: #d97706;
            color: #000;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        
        /* Timestamp */
        .timestamp {
            color: #444;
            font-size: 11px;
            text-align: right;
            margin-top: 8px;
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            background: #0d0d0d !important;
            border: 1px solid #333 !important;
            border-radius: 12px !important;
            color: #fff !important;
            padding: 12px 16px !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #d97706 !important;
            box-shadow: 0 0 0 1px #d97706 !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #d97706 0%, #b45309 100%) !important;
            color: #000 !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 32px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(217, 119, 6, 0.3) !important;
        }
        
        /* Slider */
        .stSlider > div > div > div { background: #d97706 !important; }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: #1a1a1a !important;
            border-radius: 12px !important;
        }
        
        /* Video container */
        .video-container {
            background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
            border-radius: 16px;
            padding: 16px;
            border: 1px solid #2a2a2a;
            margin: 20px 0;
        }
        .video-title {
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Speaker cards */
        .speaker-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }
        .speaker-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid #2a3f5f;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .speaker-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }
        .speaker-card.speaker-a {
            border-left: 4px solid #f472b6;
        }
        .speaker-card.speaker-b {
            border-left: 4px solid #60a5fa;
        }
        .speaker-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }
        .speaker-avatar {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            font-weight: 600;
        }
        .speaker-avatar.a { background: linear-gradient(135deg, #f472b6 0%, #db2777 100%); color: #fff; }
        .speaker-avatar.b { background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%); color: #fff; }
        .speaker-name {
            font-size: 18px;
            font-weight: 600;
            color: #fff;
        }
        .speaker-role {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .speaker-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin: 16px 0;
            padding: 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 20px;
            font-weight: 600;
            color: #fff;
        }
        .stat-label {
            font-size: 10px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .speaker-stance {
            background: rgba(217, 119, 6, 0.1);
            border: 1px solid rgba(217, 119, 6, 0.3);
            border-radius: 8px;
            padding: 10px 12px;
            margin-top: 12px;
        }
        .stance-label {
            font-size: 10px;
            color: #d97706;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .stance-text {
            font-size: 13px;
            color: #e8dcc8;
            font-style: italic;
        }
        .speaker-style {
            font-size: 11px;
            color: #666;
            margin-top: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Morning"
    elif hour < 17:
        return "Afternoon"
    else:
        return "Evening"


def log_interaction(question: str, answer: str, k: int, snippets: list[str], transcript: str):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "k": k,
        "snippets": snippets,
        "transcript": transcript,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    records = []
    with LOG_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────
# Session management (Chat 1, Chat 2, ...)
# ─────────────────────────────────────────────────────────────
def ensure_session_index():
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    if not SESSION_INDEX.exists():
        SESSION_INDEX.write_text(json.dumps({"sessions": []}, ensure_ascii=False, indent=2))


def list_sessions() -> list[dict]:
    ensure_session_index()
    try:
        data = json.loads(SESSION_INDEX.read_text())
        return data.get("sessions", [])
    except Exception:
        return []


def save_sessions_metadata(sessions: list[dict]):
    SESSION_INDEX.write_text(json.dumps({"sessions": sessions}, ensure_ascii=False, indent=2))


def session_path(session_id: str) -> Path:
    safe = session_id.strip().replace(" ", "_")
    return SESSION_DIR / f"{safe}.jsonl"


def load_session_messages(session_id: str) -> list[dict]:
    path = session_path(session_id)
    if not path.exists():
        return []
    msgs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    msgs.append(json.loads(line))
                except Exception:
                    continue
    return msgs


def append_session_message(session_id: str, msg: dict):
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    path = session_path(session_id)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def ensure_session_exists(session_id: str):
    ensure_session_index()
    sessions = list_sessions()
    if not any(s.get("id") == session_id for s in sessions):
        now = datetime.now().isoformat()
        sessions.append({"id": session_id, "title": session_id, "created_at": now, "updated_at": now})
        save_sessions_metadata(sessions)
        # Create empty session file
        session_path(session_id).touch()


@st.cache_resource
def get_clients():
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    db = PersistentClient(path=CHROMA_DIR)
    try:
        coll_f = db.get_collection(COLLECTION_FINE)
        coll_c = db.get_collection(COLLECTION_COARSE)
    except Exception:
        # If missing, attempt rebuild from default transcript.
        if DEFAULT_TRANSCRIPT.exists():
            build_index(DEFAULT_TRANSCRIPT, reset=True)
            coll_f = db.get_collection(COLLECTION_FINE)
            coll_c = db.get_collection(COLLECTION_COARSE)
        else:
            st.error("Vector index not found and default transcript is missing. Please upload a transcript or rebuild locally.")
            st.stop()
    return emb, coll_f, coll_c


def format_ts(seconds: float) -> str:
    try:
        return f"{float(seconds):.1f}s"
    except Exception:
        return str(seconds)


def speaker_display(speaker_id: str) -> str:
    """Map raw speaker id to friendly name using SPEAKER_NAMES."""
    if not speaker_id:
        return "?"
    letter = speaker_id[-1].upper()
    return SPEAKER_NAMES.get(speaker_id, SPEAKER_NAMES.get(letter, speaker_id))


def retrieve(question: str, k: int = 6, conversation_history: list = None, openai_key: str = None, model_override: str = None):
    """
    Retrieve relevant snippets and generate an answer using GPT-4o.
    Includes conversation history for context continuity.
    """
    emb_model, coll_f, coll_c = get_clients()
    q_emb = emb_model.encode([question], convert_to_numpy=True).tolist()[0]

    # Two-stage semantic zoom
    k_coarse = max(3, k)
    k_fine_per_seg = max(2, k // 2)

    # Stage 1: coarse segments
    res_c = coll_c.query(query_embeddings=[q_emb], n_results=k_coarse)
    seg_ids = []
    for meta in res_c["metadatas"][0]:
        sid = meta.get("segment_id")
        if sid:
            seg_ids.append(sid)

    # Stage 2: fine utterances within top segments
    candidates = []
    for sid in seg_ids:
        res_f = coll_f.query(
            query_embeddings=[q_emb],
            where={"segment_id": sid},
            n_results=k_fine_per_seg,
        )
        for doc, m, dist in zip(res_f["documents"][0], res_f["metadatas"][0], res_f["distances"][0]):
            candidates.append((dist, doc, m))

    candidates.sort(key=lambda x: x[0])
    top = candidates[:k] if candidates else []

    snippets = []
    for _, doc, m in top:
        raw_speaker = m.get("personId") or m.get("speaker") or "?"
        speaker = speaker_display(raw_speaker)
        snippets.append(
            f"[{speaker} @ {format_ts(m.get('startTime'))}-{format_ts(m.get('endTime'))}] {doc}"
        )
    
    # Build messages for OpenAI chat format
    system_prompt = (
        "You are an expert analyst reviewing a conversation transcript. "
        "Use the provided transcript snippets to answer questions accurately. "
        "Be concise, cite speakers by name, and reference timestamps when relevant. "
        "If the user asks for clarification, corrections, or follow-ups, use the conversation history. "
        "If the user gives feedback, adjust your response accordingly. "
        "Canonical speaker names: A=Adam, B=Tal, speaker_1=Adam, speaker_2=Tal. "
        "If asked for the names of the debaters, respond with Adam and Tal."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    if conversation_history:
        recent = conversation_history[-10:]  # Last 5 Q&A pairs
        for msg in recent:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Add current question with snippets
    user_message = (
        f"Relevant transcript snippets:\n"
        + "\n".join(snippets)
        + f"\n\nQuestion: {question}"
    )
    messages.append({"role": "user", "content": user_message})
    
    # Call OpenAI
    api_key = openai_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ OpenAI API key not set. Please add it in Settings.", snippets

    model = model_override or os.environ.get("OPENAI_MODEL") or st.session_state.get("openai_model", DEFAULT_OPENAI_MODEL)
    
    client = OpenAI(api_key=api_key)
    completion_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }

    # Prefer new param for newer models; fallback if unsupported.
    def _run_completion(use_new_param: bool):
        kwargs = completion_kwargs.copy()
        if use_new_param:
            kwargs["max_completion_tokens"] = 1000
        else:
            kwargs["max_tokens"] = 1000
        return client.chat.completions.create(**kwargs)

    try:
        response = _run_completion(use_new_param=True)
    except Exception as e:
        msg = str(e)
        if "max_completion_tokens" in msg or "unsupported_parameter" in msg:
            try:
                response = _run_completion(use_new_param=False)
            except Exception as e2:
                answer = f"⚠️ OpenAI API error: {str(e2)}"
                return answer, snippets
        else:
            answer = f"⚠️ OpenAI API error: {msg}"
            return answer, snippets

    answer = response.choices[0].message.content.strip()
    
    return answer, snippets


def main():
    inject_css()

    # Initialize session state
    ensure_session_index()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = "Chat 1"
    if "transcript_path" not in st.session_state:
        st.session_state["transcript_path"] = DEFAULT_TRANSCRIPT
    if "k_value" not in st.session_state:
        st.session_state["k_value"] = 6
    if "youtube_url" not in st.session_state:
        st.session_state["youtube_url"] = None
    if "speaker_profiles" not in st.session_state:
        st.session_state["speaker_profiles"] = None
    if "pending_question" not in st.session_state:
        st.session_state["pending_question"] = None

    # ─────────────────────────────────────────────────────────────
    # TOP BAR: Settings & Transcript dropdown
    # ─────────────────────────────────────────────────────────────
    top_col1, top_col2, top_col3 = st.columns([1.2, 2.5, 2.5])
    
    with top_col1:
        with st.popover("⚙️ Settings"):
            st.markdown("**🔑 OpenAI API Key**")
            openai_key_input = st.text_input(
                "OpenAI Key",
                value=st.session_state.get("openai_key", os.environ.get("OPENAI_API_KEY", "")),
                type="password",
                label_visibility="collapsed",
                placeholder="sk-...",
            )
            if openai_key_input:
                st.session_state["openai_key"] = openai_key_input
            
            st.divider()
            st.markdown("**🤖 OpenAI Model**")
            current_model = st.session_state.get("openai_model", DEFAULT_OPENAI_MODEL)
            model_choice = st.selectbox(
                "Model",
                options=MODEL_CHOICES,
                index=MODEL_CHOICES.index(current_model) if current_model in MODEL_CHOICES else 0,
                label_visibility="collapsed",
            )
            custom_model = st.text_input(
                "Or custom model id",
                value=current_model if current_model not in MODEL_CHOICES else "",
                placeholder="e.g., gpt-4.1-mini",
                label_visibility="collapsed",
            )
            final_model = custom_model.strip() or model_choice
            st.session_state["openai_model"] = final_model
            
            st.divider()
            st.markdown("**Context Depth**")
            st.session_state["k_value"] = st.slider(
                "k", 3, 10, st.session_state["k_value"],
                help="Number of transcript snippets to retrieve",
                label_visibility="collapsed",
            )
            st.divider()
            st.caption(f"**Model:** GPT-4o")
            st.caption(f"**Transcript:** `{st.session_state['transcript_path'].name}`")
            history = load_log()
            st.caption(f"**History:** {len(history)} conversations")
            
            col1, col2 = st.columns(2)
            if col1.button("Clear Chat", use_container_width=True):
                st.session_state["messages"] = []
                if LOG_FILE.exists():
                    LOG_FILE.unlink()
                st.rerun()
            if history:
                col2.download_button(
                    "Export",
                    data=LOG_FILE.read_text() if LOG_FILE.exists() else "",
                    file_name="chat_history.jsonl",
                    use_container_width=True,
                )

    # Session selector / new chat
    with top_col2:
        st.caption("Chat Sessions")
        sessions = list_sessions()
        session_options = [s.get("id") for s in sessions] or ["Chat 1"]
        current_session = st.selectbox(
            "Select chat",
            options=session_options,
            index=session_options.index(st.session_state["session_id"]) if st.session_state["session_id"] in session_options else 0,
            label_visibility="collapsed",
        )
        if current_session != st.session_state["session_id"]:
            st.session_state["session_id"] = current_session
            ensure_session_exists(current_session)
            st.session_state["messages"] = load_session_messages(current_session)
            st.rerun()

    with top_col3:
        st.caption("New Chat")
        new_name = st.text_input("Chat name", value="", placeholder="e.g., Chat 2", label_visibility="collapsed")
        if st.button("➕ Create Chat", use_container_width=True):
            name = new_name.strip() or f"Chat {len(session_options)+1}"
            ensure_session_exists(name)
            st.session_state["session_id"] = name
            st.session_state["messages"] = []
            st.rerun()

    # ─────────────────────────────────────────────────────────────
    # HERO: Speaker Profiles (always visible on landing)
    # ─────────────────────────────────────────────────────────────
    if st.session_state["transcript_path"].exists() and not st.session_state["messages"]:
        # Auto-load speaker stats (without LLM analysis) for quick display
        if not st.session_state.get("speaker_profiles"):
            speaker_stats = analyze_speakers(st.session_state["transcript_path"])
            if speaker_stats:
                # Quick profiles without LLM
                profiles = {}
                for spk, stats in speaker_stats.items():
                    profiles[spk] = {
                        **stats,
                        "likely_role": "participant",
                        "apparent_stance": "Click analyze for details",
                        "speaking_style": "—",
                    }
                st.session_state["speaker_profiles"] = profiles
        
        if st.session_state.get("speaker_profiles"):
            profiles = st.session_state["speaker_profiles"]
            sorted_speakers = sorted(profiles.items())
            
            # Title using native Streamlit
            st.markdown("## The Conversation")
            st.caption("Ask questions about what Adam and Tal discussed")
            st.markdown("")
            
            # Speaker cards side by side using native components
            cols = st.columns(len(sorted_speakers))
            
            for i, (spk, data) in enumerate(sorted_speakers):
                with cols[i]:
                    letter = spk[-1].upper() if spk else "?"
                    display_name = SPEAKER_NAMES.get(spk, SPEAKER_NAMES.get(letter, f"Speaker {letter}"))
                    
                    minutes = data.get("duration", 0) / 60
                    words = data.get("words", 0)
                    wpm = int(words / minutes) if minutes > 0 else 0
                    
                    role = data.get("likely_role", "participant").title()
                    stance = data.get("apparent_stance", "—")
                    
                    # Use container with border
                    with st.container(border=True):
                        # Header with avatar emoji and name
                        avatar = "🩷" if i == 0 else "💙"
                        st.markdown(f"### {avatar} {display_name}")
                        st.caption(role.upper())
                        
                        # Stats in 3 columns
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Minutes", f"{minutes:.1f}")
                        m2.metric("Words", f"{words:,}")
                        m3.metric("WPM", wpm)
                        
                        # Stance
                        st.info(f"**Stance:** _{stance}_")
            
            # Analyze button if not yet analyzed
            if any(p.get("speaking_style") == "—" for p in profiles.values()):
                st.markdown("")
                if st.button("🔍 Analyze Speakers with AI", use_container_width=True):
                    with st.spinner("Analyzing speakers..."):
                        speaker_stats = analyze_speakers(st.session_state["transcript_path"])
                        new_profiles = {}
                        for spk, stats in speaker_stats.items():
                            analysis = identify_speaker_with_llm(spk, stats.get("quotes", []), st.session_state.get("openai_key"))
                            new_profiles[spk] = {**stats, **analysis}
                        st.session_state["speaker_profiles"] = new_profiles
                    st.rerun()

    # ─────────────────────────────────────────────────────────────
    # CHAT: Display conversation history (using native Streamlit)
    # ─────────────────────────────────────────────────────────────
    if st.session_state["messages"]:
        st.markdown("---")
        
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])
                    
                    # Evidence snippets
                    if msg.get("snippets"):
                        with st.expander(f"📎 Evidence (k={msg.get('k', '?')})"):
                            for snip in msg["snippets"][:3]:
                                st.caption(f"• {snip[:150]}..." if len(snip) > 150 else f"• {snip}")
                    
                    if msg.get("timestamp"):
                        st.caption(f"_{msg['timestamp']}_")

    # ─────────────────────────────────────────────────────────────
    # CHAT INPUT: Using chat_input for Enter key support
    # ─────────────────────────────────────────────────────────────
    
    question = st.chat_input("Ask about the conversation (e.g., What did Adam and Tal disagree about?)")
    
    if question and question.strip():
        st.session_state["messages"].append({"role": "user", "content": question.strip()})
        append_session_message(st.session_state["session_id"], {"role": "user", "content": question.strip()})
        
        with st.spinner("Thinking..."):
            # Pass conversation history (excluding the just-added message)
            history = st.session_state["messages"][:-1]
            answer, snippets = retrieve(
                question.strip(),
                k=st.session_state["k_value"],
                conversation_history=history,
                openai_key=st.session_state.get("openai_key"),
                model_override=st.session_state.get("openai_model", DEFAULT_OPENAI_MODEL),
            )
        
        timestamp = datetime.now().strftime("%H:%M")
        assistant_msg = {
            "role": "assistant",
            "content": answer,
            "snippets": snippets,
            "k": st.session_state["k_value"],
            "timestamp": timestamp,
        }
        st.session_state["messages"].append(assistant_msg)
        append_session_message(st.session_state["session_id"], assistant_msg)
        
        log_interaction(
            question=question.strip(),
            answer=answer,
            k=st.session_state["k_value"],
            snippets=snippets,
            transcript=str(st.session_state["transcript_path"]),
        )
        
        st.rerun()

    # ─────────────────────────────────────────────────────────────
    # BOTTOM: YouTube Loader (separate section)
    # ─────────────────────────────────────────────────────────────
    st.markdown("")
    st.divider()
    
    with st.expander("🎬 Load a different YouTube video"):
        yt_col1, yt_col2 = st.columns([3, 1])
        
        with yt_col1:
            url = st.text_input(
                "YouTube URL",
                placeholder="https://youtube.com/watch?v=...",
                label_visibility="collapsed",
            )
        with yt_col2:
            speakers = st.number_input("Speakers", min_value=1, max_value=10, value=2, label_visibility="collapsed")
        
        api_key = os.environ.get("ASSEMBLYAI_API_KEY") or st.text_input(
            "AssemblyAI API Key",
            type="password",
            placeholder="Enter your AssemblyAI key",
        )
        
        if st.button("🚀 Create New Chatbot", use_container_width=True):
            if url.strip() and api_key:
                with st.spinner("Downloading and transcribing video..."):
                    out_path = Path("data/session.jsonl")
                    processed = process_youtube_to_jsonl(
                        url.strip(), out_path, api_key=api_key,
                        speakers_expected=int(speakers),
                    )
                    build_index(processed, reset=True)
                    st.session_state["transcript_path"] = processed
                    st.session_state["youtube_url"] = url.strip()
                    st.session_state["messages"] = []
                    st.session_state["speaker_profiles"] = None
                
                st.success("Ready! Refresh to see the new conversation.")
                st.rerun()
            else:
                st.error("Please provide both a YouTube URL and API key.")


# Streamlit runs this file as __main__, but call main() unconditionally for clarity.
main()
