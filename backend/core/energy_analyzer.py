"""
Energy Analyzer - Pacing Analysis for Audio Segments
Analyzes word rate, pauses, and energy levels for B-roll matching
"""

from typing import Dict, List


def calculate_word_rate(word_segments: List[Dict]) -> float:
    """Calculate average speaking rate in words per second"""
    if len(word_segments) < 2:
        return 0.0
    total_words = len(word_segments)
    total_duration = word_segments[-1]["end"] - word_segments[0]["start"]
    return total_words / total_duration if total_duration > 0 else 0.0


def analyze_pauses(word_segments: List[Dict], threshold: float = 0.5) -> Dict:
    """Analyze pause patterns in word segments"""
    pauses = []
    for i, word in enumerate(word_segments[:-1]):
        gap = word_segments[i + 1]["start"] - word["end"]
        if gap > threshold:
            pauses.append(gap)
    return {
        "pause_count": len(pauses),
        "total_pause_duration": sum(pauses),
        "avg_pause_duration": sum(pauses) / len(pauses) if pauses else 0.0,
    }


def classify_energy(word_rate: float, pause_analysis: Dict) -> str:
    """Classify segment energy level: low, medium, high"""
    if word_rate > 3.0 and pause_analysis["pause_count"] < 2:
        return "high"
    elif word_rate < 1.5 and pause_analysis["pause_count"] > 4:
        return "low"
    else:
        return "medium"


def tag_segment_energy(segment: Dict, word_segments: List[Dict]) -> Dict:
    """Add energy metadata to segment based on word segments within time range"""
    seg_start = segment["start"]
    seg_end = segment["end"]

    segment_words = [
        w for w in word_segments
        if w["start"] >= seg_start and w["end"] <= seg_end
    ]

    if not segment_words:
        segment_words = [
            w for w in word_segments
            if w["start"] >= seg_start and w["start"] < seg_end
        ]

    if len(segment_words) < 2:
        word_rate = 0.0
        pause_analysis = {"pause_count": 0, "total_pause_duration": 0.0, "avg_pause_duration": 0.0}
    else:
        word_rate = calculate_word_rate(segment_words)
        pause_analysis = analyze_pauses(segment_words)

    energy_level = classify_energy(word_rate, pause_analysis)

    return {
        **segment,
        "energy_level": energy_level,
        "word_rate": round(word_rate, 2),
        "pause_count": pause_analysis["pause_count"],
    }


def enrich_segments_with_energy(
    segments: List[Dict],
    word_segments: List[Dict]
) -> List[Dict]:
    """Enrich all segments with energy analysis metadata"""
    return [tag_segment_energy(seg, word_segments) for seg in segments]
