import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize(records: Iterable[dict]) -> Dict:
    per_speaker: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"durations": [], "words": []}
    )

    total_duration = 0.0
    total_words = 0
    start_min = None
    end_max = None
    total_turns = 0

    for rec in records:
        start = float(rec.get("startTime", 0))
        end = float(rec.get("endTime", start))
        dur = max(0.0, end - start)
        speaker = rec.get("personId", "unknown")
        text = rec.get("text", "") or ""
        words = len(text.split())

        total_turns += 1
        total_duration += dur
        total_words += words

        per_speaker[speaker]["durations"].append(dur)
        per_speaker[speaker]["words"].append(words)

        start_min = start if start_min is None else min(start_min, start)
        end_max = end if end_max is None else max(end_max, end)

    overall_span = (end_max - start_min) if start_min is not None else 0.0

    def speaker_stats(durations: List[float], words: List[int]) -> Dict:
        return {
            "turns": len(durations),
            "talk_time_sec": sum(durations),
            "avg_turn_sec": mean(durations) if durations else 0.0,
            "median_turn_sec": median(durations) if durations else 0.0,
            "longest_turn_sec": max(durations) if durations else 0.0,
            "words": sum(words),
            "avg_words_per_turn": (sum(words) / len(words)) if durations else 0.0,
            "words_per_minute": (
                (sum(words) / (sum(durations) / 60.0)) if sum(durations) > 0 else 0.0
            ),
        }

    speaker_summary = {
        speaker: speaker_stats(data["durations"], data["words"])
        for speaker, data in per_speaker.items()
    }

    overall = {
        "turns": total_turns,
        "talk_time_sec": total_duration,
        "conversation_span_sec": overall_span,
        "avg_turn_sec": (total_duration / total_turns) if total_turns else 0.0,
        "median_turn_sec": median(
            [dur for data in per_speaker.values() for dur in data["durations"]]
        )
        if total_turns
        else 0.0,
        "total_words": total_words,
        "words_per_minute": (total_words / (total_duration / 60.0))
        if total_duration > 0
        else 0.0,
    }

    return {"overall": overall, "per_speaker": speaker_summary}


def format_table(stats: Dict) -> str:
    lines = []
    per_speaker = stats["per_speaker"]
    overall = stats["overall"]

    lines.append("=== Overall ===")
    lines.append(f"Turns: {overall['turns']}")
    lines.append(f"Talk time (sec): {overall['talk_time_sec']:.2f}")
    lines.append(f"Conversation span (sec): {overall['conversation_span_sec']:.2f}")
    lines.append(f"Avg turn (sec): {overall['avg_turn_sec']:.2f}")
    lines.append(f"Median turn (sec): {overall['median_turn_sec']:.2f}")
    lines.append(f"Total words: {overall['total_words']}")
    lines.append(f"Words per minute: {overall['words_per_minute']:.2f}")
    lines.append("")
    lines.append("=== By speaker ===")
    header = (
        f"{'speaker':<15}{'turns':>8}{'talk_sec':>12}{'avg_sec':>10}"
        f"{'median_sec':>12}{'longest':>10}{'words':>10}{'wpm':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for speaker, s in sorted(per_speaker.items()):
        lines.append(
            f"{speaker:<15}{s['turns']:>8}{s['talk_time_sec']:>12.2f}"
            f"{s['avg_turn_sec']:>10.2f}{s['median_turn_sec']:>12.2f}"
            f"{s['longest_turn_sec']:>10.2f}{s['words']:>10}"
            f"{s['words_per_minute']:>8.1f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute descriptive stats for a transcript JSONL."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/h9g9SnPsN9s.formatted.jsonl"),
        help="Path to transcript JSONL with startTime, endTime, text, personId fields.",
    )
    args = parser.parse_args()

    if not args.file.exists():
        raise SystemExit(f"File not found: {args.file}")

    stats = summarize(load_jsonl(args.file))
    print(format_table(stats))


if __name__ == "__main__":
    main()
