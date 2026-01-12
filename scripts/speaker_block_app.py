"""
CLI utility to analyze speaking time and continuous blocks per speaker
from a transcript JSONL file. Designed for files with fields:
- speaker or personId
- start or startTime
- end or endTime
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> Iterable[dict]:
    """Yield JSON objects from a JSONL file, skipping blank lines."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_record(rec: dict) -> Tuple[str, float, float]:
    """Return (speaker, start, end) using common key variants."""
    speaker = rec.get("speaker") or rec.get("personId") or "unknown"
    start = rec.get("start")
    end = rec.get("end")
    if start is None:
        start = rec.get("startTime", 0.0)
    if end is None:
        end = rec.get("endTime", start if start is not None else 0.0)
    start = float(start)
    end = float(end)
    return speaker, start, end


def compute_blocks(records: Iterable[dict]) -> List[Dict]:
    """
    Collapse consecutive turns by the same speaker into speaking blocks.
    Each block is a dict with speaker, start, end, dur.
    """
    blocks: List[Dict] = []
    current = None

    for rec in records:
        speaker, start, end = normalize_record(rec)
        if current is None:
            current = {"speaker": speaker, "start": start, "end": end}
            continue

        if speaker == current["speaker"]:
            # Extend current block
            if end > current["end"]:
                current["end"] = end
        else:
            current["dur"] = max(0.0, current["end"] - current["start"])
            blocks.append(current)
            current = {"speaker": speaker, "start": start, "end": end}

    if current:
        current["dur"] = max(0.0, current["end"] - current["start"])
        blocks.append(current)

    return blocks


def describe_blocks(blocks: List[Dict]) -> Dict[str, Dict]:
    """Return per-speaker descriptive stats over block durations."""
    per_speaker: Dict[str, List[float]] = defaultdict(list)
    for blk in blocks:
        per_speaker[blk["speaker"]].append(blk["dur"])

    def p90(vals: List[float]) -> float:
        if not vals:
            return 0.0
        vals_sorted = sorted(vals)
        idx = max(0, int(0.9 * len(vals_sorted)) - 1)
        return vals_sorted[idx]

    stats: Dict[str, Dict] = {}
    for speaker, durations in per_speaker.items():
        stats[speaker] = {
            "blocks": len(durations),
            "total_sec": sum(durations),
            "mean_sec": mean(durations) if durations else 0.0,
            "median_sec": median(durations) if durations else 0.0,
            "p90_sec": p90(durations),
            "max_sec": max(durations) if durations else 0.0,
        }
    return stats


def format_report(blocks: List[Dict]) -> str:
    """Render a human-readable report."""
    lines: List[str] = []
    block_stats = describe_blocks(blocks)
    overall_total = sum(blk["dur"] for blk in blocks)

    lines.append(f"Blocks identified: {len(blocks)}")
    lines.append(f"Speaker changes: {max(0, len(blocks) - 1)}")
    lines.append("")
    lines.append("=== By speaker (continuous blocks) ===")
    header = (
        f"{'speaker':<15}{'blocks':>8}{'total_sec':>12}"
        f"{'mean_sec':>10}{'median':>10}{'p90':>8}{'max':>8}{'share':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for speaker, stats in sorted(block_stats.items()):
        share = (
            (stats["total_sec"] / overall_total * 100) if overall_total else 0.0
        )
        lines.append(
            f"{speaker:<15}{stats['blocks']:>8}"
            f"{stats['total_sec']:>12.2f}{stats['mean_sec']:>10.2f}"
            f"{stats['median_sec']:>10.2f}{stats['p90_sec']:>8.2f}"
            f"{stats['max_sec']:>8.2f}{share:>8.1f}%"
        )

    lines.append("")
    lines.append("=== Top 5 longest blocks per speaker ===")
    for speaker in sorted(block_stats.keys()):
        top = sorted(
            [b for b in blocks if b["speaker"] == speaker],
            key=lambda x: x["dur"],
            reverse=True,
        )[:5]
        lines.append(f"{speaker}:")
        if not top:
            lines.append("  (none)")
            continue
        for blk in top:
            lines.append(
                f"  {blk['start']:.2f}->{blk['end']:.2f} sec (dur {blk['dur']:.2f})"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze continuous speaking blocks per speaker in a transcript JSONL."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/h9g9SnPsN9s.formatted.jsonl"),
        help="Path to transcript JSONL (fields: speaker/personId, start/startTime, end/endTime).",
    )
    args = parser.parse_args()

    if not args.file.exists():
        raise SystemExit(f"File not found: {args.file}")

    records = list(load_jsonl(args.file))
    if not records:
        raise SystemExit("No records found.")

    blocks = compute_blocks(records)
    print(format_report(blocks))


if __name__ == "__main__":
    main()
