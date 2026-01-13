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
            List of segments with text, start, end, and duration
        """
        import os
        import json
        import requests
        
        # Get full transcript
        word_segments = aligned_result.get("word_segments", [])
        raw_segments = aligned_result.get("segments", [])
        
        # Build full transcript with timestamps
        full_text = " ".join([w["word"] for w in word_segments]) if word_segments else " ".join([s["text"] for s in raw_segments])
        
        if not full_text.strip():
            return []
        
        # Get video duration
        if word_segments:
            video_duration = word_segments[-1]["end"]
        elif raw_segments:
            video_duration = raw_segments[-1]["end"]
        else:
            video_duration = 30.0  # Default
        
        # Ask LLM to identify segment boundaries
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ No GOOGLE_API_KEY, using fallback segmentation")
            return self._fallback_segmentation(aligned_result, min_duration, max_duration)
        
        prompt = f"""You are a video editor identifying B-Roll insertion points.

TRANSCRIPT: "{full_text}"
VIDEO DURATION: {video_duration:.1f} seconds

TASK: Identify 2-6 natural "cut points" where B-Roll would enhance the narrative.

Think about:
1. Topic changes ("First... then... finally...")
2. Key visual moments (describing actions, places, objects)
3. Emotional shifts
4. Pauses or transitions

OUTPUT FORMAT (JSON only):
{{
  "segments": [
    {{"text": "first part of transcript...", "reason": "why B-Roll here"}},
    {{"text": "second part...", "reason": "why B-Roll here"}}
  ]
}}

RULES:
- Each segment should be roughly 3-15 seconds of speech
- Use EXACT words from the transcript
- Cover the ENTIRE transcript (no gaps)
- 2-6 segments total"""

        try:
            endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
            headers = {"Content-Type": "application/json"}
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 500}
            }
            
            response = requests.post(f"{endpoint}?key={api_key}", headers=headers, json=payload, timeout=30)
            
            if response.status_code != 200:
                print(f"⚠️ LLM segmentation failed: {response.status_code}")
                return self._fallback_segmentation(aligned_result, min_duration, max_duration)
            
            result_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^{}]*"segments"[^{}]*\[.*?\]\s*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(result_text)
            
            llm_segments = result.get("segments", [])
            
            if not llm_segments:
                return self._fallback_segmentation(aligned_result, min_duration, max_duration)
            
            # Map LLM segments to timestamps
            final_segments = []
            word_index = 0
            
            for seg_data in llm_segments:
                seg_text = seg_data.get("text", "").strip()
                if not seg_text:
                    continue
                
                # Find matching words
                seg_words = seg_text.lower().split()[:5]  # Match first 5 words
                
                # Find start position
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
                        
                        # Check if we've matched enough
                        if len(matched_text.split()) >= len(seg_text.split()) * 0.8:
                            word_index = i + 1
                            break
                
                if start_time is not None and end_time is not None:
                    final_segments.append({
                        "text": matched_text.strip(),
                        "start": start_time,
                        "end": end_time,
                        "duration": end_time - start_time,
                        "llm_reason": seg_data.get("reason", "")
                    })
            
            print(f"✓ LLM created {len(final_segments)} semantic segments")
            for i, seg in enumerate(final_segments):
                print(f"   Segment {i+1}: {seg['text'][:40]}... ({seg['duration']:.1f}s)")
            
            return final_segments if final_segments else self._fallback_segmentation(aligned_result, min_duration, max_duration)
            
        except Exception as e:
            print(f"⚠️ LLM segmentation error: {e}")
            return self._fallback_segmentation(aligned_result, min_duration, max_duration)
    
    def _fallback_segmentation(
        self,
        aligned_result: Dict,
        min_duration: float = 3.0,
        max_duration: float = 10.0
    ) -> List[Dict]:
        """Simple time-based fallback if LLM fails"""
        segments = []
        current = {"text": "", "start": None, "end": None}
        word_segments = aligned_result.get("word_segments", [])
        
        if not word_segments:
            for seg in aligned_result.get("segments", []):
                segments.append({
                    "text": seg["text"].strip(),
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": seg["end"] - seg["start"]
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
                segments.append({
                    "text": current["text"].strip(),
                    "start": current["start"],
                    "end": current["end"],
                    "duration": duration
                })
                current = {"text": "", "start": None, "end": None}
        
        if current["text"].strip():
            segments.append({
                "text": current["text"].strip(),
                "start": current["start"],
                "end": current["end"],
                "duration": current["end"] - current["start"]
            })
        
        return segments
    
    def cleanup(self):
        """Cleanup (no GPU memory to free with Vertex AI!)"""
        print("✓ Audio sensor cleanup complete (no resources to free)")
