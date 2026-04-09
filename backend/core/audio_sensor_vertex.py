"""
Audio Sensor Module - Vertex AI Speech-to-Text Version
Extracts word-level transcription using Google Cloud Speech-to-Text V2
"""

from google.cloud import speech_v2
from google.cloud.speech_v2 import types
from pathlib import Path
from typing import Dict, List
import subprocess
import os

DEFAULT_FILLERS = {"um", "uh", "er", "ah", "like", "you know", "well"}


def detect_pauses(word_segments: List[Dict], pause_threshold: float = 2.0) -> List[Dict]:
    """Identify natural cut points where speaker pauses > threshold seconds."""
    if len(word_segments) < 2:
        return []
    
    pauses = []
    for i, word in enumerate(word_segments[:-1]):
        gap = word_segments[i + 1]["start"] - word["end"]
        if gap > pause_threshold:
            pauses.append({
                "index": i,
                "start": word["end"],
                "end": word_segments[i + 1]["start"],
                "duration": gap
            })
    return pauses


def count_pauses_in_range(word_segments: List[Dict], pauses: List[Dict], start: float, end: float) -> int:
    """Count pauses within a time range."""
    return sum(1 for p in pauses if start <= p["start"] <= end)


def filter_low_confidence(
    word_segments: List[Dict],
    min_confidence: float = 0.7,
    fillers: set = None
) -> List[Dict]:
    """Filter words with confidence < min_confidence and filler words."""
    fillers = fillers or DEFAULT_FILLERS
    return [
        w for w in word_segments
        if w.get("confidence", 1.0) >= min_confidence
        and w["word"].lower().strip() not in fillers
    ]


def calculate_avg_confidence(word_segments: List[Dict]) -> float:
    """Calculate average confidence of word segments."""
    if not word_segments:
        return 1.0
    return sum(w.get("confidence", 1.0) for w in word_segments) / len(word_segments)


def estimate_energy_level(word_segments: List[Dict]) -> str:
    """Estimate speech energy level based on pace (words per second)."""
    if len(word_segments) < 2:
        return "medium"
    
    total_duration = word_segments[-1]["end"] - word_segments[0]["start"]
    if total_duration <= 0:
        return "medium"
    
    words_per_second = len(word_segments) / total_duration
    
    if words_per_second < 1.5:
        return "low"
    elif words_per_second > 3.0:
        return "high"
    return "medium"


def extract_topics_from_segments(segments: List[Dict]) -> List[str]:
    """Extract top topics from segment texts (simple keyword approach)."""
    topic_keywords = {
        "technical": {"code", "programming", "software", "api", "function", "system"},
        "personal": {"i", "me", "my", "we", "our", "family", "home"},
        "business": {"company", "revenue", "customer", "market", "sales", "product"},
        "creative": {"design", "art", "creative", "music", "video", "story"},
        "process": {"step", "process", "method", "approach", "workflow", "pipeline"},
    }
    
    all_text = " ".join(seg.get("text", "").lower() for seg in segments)
    detected = []
    
    for topic, keywords in topic_keywords.items():
        if any(kw in all_text for kw in keywords):
            detected.append(topic)
    
    return detected[:5]


class AudioSensor:
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize Audio Sensor with Vertex AI Speech-to-Text
        
        Benefits over WhisperX:
        - No PyTorch (eliminates compatibility issues)
        - Same GCP authentication (already setup)
        - Native word-level timestamps
        - Auto language detection
        - Handles Hinglish well
        
        Args:
            project_id: GCP project ID
            location: GCP region (default: us-central1)
        """
        self.project_id = project_id
        self.location = location
        self.client = speech_v2.SpeechClient()
        
        print(f"✓ Vertex AI Speech-to-Text initialized (project: {project_id})")
    
    def extract_audio(self, video_path: Path, output_path: Path) -> Path:
        """Extract audio using FFmpeg at 16kHz mono"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn",  # No video
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",  # Mono
            "-acodec", "pcm_s16le",  # Linear PCM
            "-f", "wav",  # WAV format for Speech-to-Text
            "-y",  # Overwrite
            str(output_path)
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    
    def transcribe_and_align(self, audio_path: Path) -> Dict:
        """
        Get word-level timestamps using Vertex AI Speech-to-Text V2
        
        Args:
            audio_path: Path to audio file (WAV format)
        
        Returns:
            Dict with 'word_segments' containing word-level data
        """
        print("Transcribing audio with Vertex AI Speech-to-Text...")
        
        # Read audio file
        with open(audio_path, "rb") as f:
            audio_content = f.read()
        
        # Configure recognition
        config = types.RecognitionConfig(
            auto_decoding_config=types.AutoDetectDecodingConfig(),
            language_codes=["en-US", "hi-IN"],  # English + Hindi for Hinglish
            model="latest_long",  # Best model for longer audio
            features=types.RecognitionFeatures(
                enable_word_time_offsets=True,  # Word-level timestamps
                enable_automatic_punctuation=True,
            ),
        )
        
        # Create request (use global location for Speech-to-Text V2)
        request = types.RecognizeRequest(
            recognizer=f"projects/{self.project_id}/locations/global/recognizers/_",
            config=config,
            content=audio_content,
        )
        
        # Recognize
        response = self.client.recognize(request=request)
        
        # Extract word segments
        word_segments = []
        segments = []
        
        for result in response.results:
            if not result.alternatives:
                continue
            
            alternative = result.alternatives[0]
            
            # Segment-level
            segments.append({
                "text": alternative.transcript,
                "start": alternative.words[0].start_offset.total_seconds() if alternative.words else 0,
                "end": alternative.words[-1].end_offset.total_seconds() if alternative.words else 0
            })
            
            # Word-level
            for word_info in alternative.words:
                word_segments.append({
                    "word": word_info.word,
                    "start": word_info.start_offset.total_seconds(),
                    "end": word_info.end_offset.total_seconds(),
                    "confidence": alternative.confidence if hasattr(alternative, 'confidence') else 1.0
                })
        
        print(f"✓ Transcribed {len(word_segments)} words in {len(segments)} segments")
        
        return {
            "segments": segments,
            "word_segments": word_segments,
            "language": "en"  # Vertex AI auto-detects
        }
    
    def create_semantic_segments(
        self,
        aligned_result: Dict,
        min_duration: float = 2.0,
        max_duration: float = 15.0
    ) -> List[Dict]:
        """
        LLM-Driven Semantic Segmentation
        
        Instead of arbitrary time-based splits, uses Gemini to identify
        natural B-Roll insertion points based on narrative meaning.
        
        Args:
            aligned_result: Output from transcribe_and_align()
            min_duration: Minimum segment length (soft constraint)
            max_duration: Maximum segment length (hard constraint for API limits)
        
        Returns:
            List of segments with text, start, end, duration, and metadata
        """
        import json
        import re
        import requests
        
        word_segments = aligned_result.get("word_segments", [])
        raw_segments = aligned_result.get("segments", [])
        
        full_text = " ".join([w["word"] for w in word_segments]) if word_segments else " ".join([s["text"] for s in raw_segments])
        
        if not full_text.strip():
            return []
        
        video_duration = (
            word_segments[-1]["end"] if word_segments
            else raw_segments[-1]["end"] if raw_segments
            else 30.0
        )
        
        pauses = detect_pauses(word_segments)
        pause_info = f"\nSIGNIFICANT PAUSES: {len(pauses)} pauses > 2s detected at times: {[round(p['start'], 1) for p in pauses]}" if pauses else ""
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ No GOOGLE_API_KEY, using fallback segmentation")
            return self._fallback_segmentation(aligned_result, min_duration, max_duration)
        
        prompt = f"""You are an expert video editor analyzing speech content.

TRANSCRIPT: "{full_text}"
VIDEO DURATION: {video_duration:.1f} seconds{pause_info}

TASK: Identify 2-6 natural segments where B-Roll would enhance the narrative.

CHAIN-OF-THOUGHT ANALYSIS:
1. NARRATIVE STRUCTURE: Identify the overall story arc (setup, development, conclusion)
2. TOPIC CHANGES: Note distinct subject matter shifts (technical topic vs personal story)
3. VISUAL MOMENTS: Look for descriptions of actions, places, objects, people
4. ENERGY LEVELS: Identify transitions between fast-paced explanation and slow reflection
5. PAUSE AWARENESS: Use natural pauses as potential segment boundaries

OUTPUT FORMAT (JSON only):
{{
  "segments": [
    {{
      "text": "exact words from transcript...",
      "reason": "narrative purpose (1 sentence)",
      "topic": "main topic category",
      "energy": "high/medium/low"
    }}
  ]
}}

RULES:
- Use EXACT words from transcript (preserve punctuation)
- Cover ENTIRE transcript (no gaps)
- 2-6 segments, each 3-15 seconds of speech
- Include transition words naturally"""

        try:
            endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent"
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800}
            }
            
            response = requests.post(f"{endpoint}?key={api_key}", headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                print(f"⚠️ LLM segmentation failed: {response.status_code}")
                return self._fallback_segmentation(aligned_result, min_duration, max_duration)
            
            result_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            json_match = re.search(r'\{[^{}]*"segments"[^{}]*\[.*?\]\s*\}', result_text, re.DOTALL)
            result = json.loads(json_match.group()) if json_match else json.loads(result_text)
            
            llm_segments = result.get("segments", [])
            
            if not llm_segments:
                return self._fallback_segmentation(aligned_result, min_duration, max_duration)
            
            final_segments = []
            word_index = 0
            
            for seg_data in llm_segments:
                seg_text = seg_data.get("text", "").strip()
                if not seg_text:
                    continue
                
                seg_words = seg_text.lower().split()[:5]
                start_time = None
                end_time = None
                matched_text = ""
                
                for i, w in enumerate(word_segments[word_index:], start=word_index):
                    if start_time is None and w["word"].lower() in seg_words[0]:
                        start_time = w["start"]
                        matched_text = w["word"]
                    elif start_time is not None:
                        matched_text += " " + w["word"]
                        end_time = w["end"]
                        
                        if len(matched_text.split()) >= len(seg_text.split()) * 0.8:
                            word_index = i + 1
                            break
                
                if start_time is not None and end_time is not None:
                    seg_word_range = [w for w in word_segments if start_time <= w["start"] <= end_time]
                    pause_count = count_pauses_in_range(word_segments, pauses, start_time, end_time)
                    avg_conf = calculate_avg_confidence(seg_word_range)
                    topics = extract_topics_from_segments([{"text": seg_data.get("topic", ""), **seg_data}])
                    
                    final_segments.append({
                        "text": matched_text.strip(),
                        "start": start_time,
                        "end": end_time,
                        "duration": end_time - start_time,
                        "llm_reason": seg_data.get("reason", ""),
                        "pause_count": pause_count,
                        "avg_confidence": round(avg_conf, 2),
                        "topics": topics or [seg_data.get("topic", "general")],
                        "energy_level": seg_data.get("energy", estimate_energy_level(seg_word_range))
                    })
            
            if not final_segments:
                return self._fallback_segmentation(aligned_result, min_duration, max_duration)
            
            print(f"✓ LLM created {len(final_segments)} semantic segments with metadata")
            for i, seg in enumerate(final_segments):
                print(f"   Segment {i+1}: {seg['text'][:40]}... ({seg['duration']:.1f}s, {seg['energy_level']})")
            
            return final_segments
            
        except Exception as e:
            print(f"⚠️ LLM segmentation error: {e}")
            return self._fallback_segmentation(aligned_result, min_duration, max_duration)
    
    def _fallback_segmentation(
        self,
        aligned_result: Dict,
        min_duration: float = 3.0,
        max_duration: float = 10.0
    ) -> List[Dict]:
        """Simple time-based fallback if LLM fails."""
        segments = []
        current = {"text": "", "start": None, "end": None}
        word_segments = aligned_result.get("word_segments", [])
        pauses = detect_pauses(word_segments)
        
        if not word_segments:
            for seg in aligned_result.get("segments", []):
                segments.append({
                    "text": seg["text"].strip(),
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": seg["end"] - seg["start"],
                    "pause_count": 0,
                    "avg_confidence": 1.0,
                    "topics": ["general"],
                    "energy_level": "medium"
                })
            return segments
        
        for word in word_segments:
            if current["start"] is None:
                current["start"] = word["start"]
            
            current["text"] += " " + word["word"]
            current["end"] = word["end"]
            
            duration = current["end"] - current["start"]
            is_sentence_end = word["word"].rstrip().endswith((".", "!", "?"))
            
            if (is_sentence_end and duration >= min_duration) or duration >= max_duration:
                seg_start = current["start"]
                seg_end = current["end"]
                seg_words = [w for w in word_segments if seg_start <= w["start"] <= seg_end]
                
                segments.append({
                    "text": current["text"].strip(),
                    "start": seg_start,
                    "end": seg_end,
                    "duration": duration,
                    "pause_count": count_pauses_in_range(word_segments, pauses, seg_start, seg_end),
                    "avg_confidence": calculate_avg_confidence(seg_words),
                    "topics": extract_topics_from_segments([current]),
                    "energy_level": estimate_energy_level(seg_words)
                })
                current = {"text": "", "start": None, "end": None}
        
        if current["text"].strip():
            seg_start = current["start"]
            seg_end = current["end"]
            seg_words = [w for w in word_segments if seg_start <= w["start"] <= seg_end]
            
            segments.append({
                "text": current["text"].strip(),
                "start": seg_start,
                "end": seg_end,
                "duration": seg_end - seg_start,
                "pause_count": count_pauses_in_range(word_segments, pauses, seg_start, seg_end),
                "avg_confidence": calculate_avg_confidence(seg_words),
                "topics": extract_topics_from_segments([current]),
                "energy_level": estimate_energy_level(seg_words)
            })
        
        return segments
    
    def cleanup(self):
        """Cleanup (no GPU memory to free with Vertex AI!)"""
        print("✓ Audio sensor cleanup complete (no resources to free)")
