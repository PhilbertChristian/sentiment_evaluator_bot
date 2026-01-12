import tempfile
from pathlib import Path
from typing import Optional

import json
import assemblyai as aai
from yt_dlp import YoutubeDL


def download_audio(url: str, out_dir: Path) -> Path:
    """Download best available audio stream for a YouTube URL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = ydl.prepare_filename(info)
    return Path(downloaded)


def transcript_to_jsonl(transcript: aai.Transcript, out_path: Path) -> None:
    """Write AssemblyAI transcript utterances to JSONL (start/end in seconds)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, utt in enumerate(transcript.utterances or [], start=1):
            # AssemblyAI gives ms; convert to seconds float
            start_s = round((utt.start or 0) / 1000, 3)
            end_s = round((utt.end or 0) / 1000, 3)
            record = {
                "utteranceId": idx,
                "personId": utt.speaker or "unknown",
                "startTime": start_s,
                "endTime": end_s,
                "text": utt.text.strip(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_youtube_to_jsonl(
    url: str,
    out_path: Path,
    *,
    api_key: str,
    speakers_expected: Optional[int] = None,
) -> Path:
    """Download YouTube audio, transcribe with diarization, and write JSONL."""
    aai.settings.api_key = api_key
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=speakers_expected,
    )
    transcriber = aai.Transcriber(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = download_audio(url, Path(tmpdir))
        transcript = transcriber.transcribe(str(audio_path))
        if transcript.status == "error":
            raise RuntimeError(f"Transcription failed: {transcript.error}")
        transcript_to_jsonl(transcript, out_path)
        return out_path
