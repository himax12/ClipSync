"""
Autonomous Multimodal Video Editor (Revamped)
ONE VLM call that returns TIMESTAMPS directly - no text matching needed.

Key principles:
1. NO translation (VLM can read any language)
2. VLM returns timestamps, not segment text
3. Dynamic cuts (VLM decides how many, not us)
4. 1 frame per B-Roll clip (sufficient for VLM)
"""

import os
import json
import base64
import requests
from typing import List, Dict, Optional
from pathlib import Path


class AutonomousEditor:
    """
    Simple VLM Video Editor.
    
    Input: Transcript + B-Roll frame images + video duration
    Output: List of cuts with timestamps and clip numbers
    
    ONE API call, NO text matching, works with ANY language.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required")
        
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        print("✓ VLM Editor initialized")
    
    def create_timeline(
        self,
        transcript: str,
        word_timestamps: List[Dict],
        available_clips: List[Dict]
    ) -> List[Dict]:
        """
        ONE API CALL to get all B-Roll decisions.
        
        VLM returns timestamps directly - no text matching needed!
        """
        # Get video duration
        if word_timestamps:
            video_duration = word_timestamps[-1]["end"]
        else:
            video_duration = len(transcript.split()) / 2.5  # Estimate
        
        num_clips = len(available_clips)
        
        print(f"\n🧠 VLM Analysis:")
        print(f"   Duration: {video_duration:.1f}s")
        print(f"   Transcript: {len(transcript.split())} words")
        print(f"   B-Roll clips: {num_clips}")
        
        # Build multimodal prompt
        parts = self._build_prompt(transcript, available_clips, video_duration)
        
        try:
            # SINGLE API CALL
            response = self._call_vlm(parts)
            
            # Parse timestamps directly
            timeline = self._parse_response(response, available_clips, word_timestamps)
            
            print(f"\n✓ VLM returned {len(timeline)} cuts:")
            for i, cut in enumerate(timeline):
                print(f"   [{i+1}] {cut['aroll_start']:.1f}s-{cut['aroll_end']:.1f}s → {cut['broll_name']}")
            
            return timeline
            
        except Exception as e:
            print(f"⚠️ VLM error: {e}")
            # Fallback: single segment with first clip
            if available_clips and word_timestamps:
                return [{
                    "aroll_start": word_timestamps[0]["start"],
                    "aroll_end": word_timestamps[-1]["end"],
                    "aroll_text": transcript[:50] + "...",
                    "broll_clip": available_clips[0].get("path"),
                    "broll_id": 0,
                    "broll_name": available_clips[0].get("name", "clip_1"),
                    "duration": video_duration,
                    "llm_reason": "Fallback"
                }]
            return []
    
    def _build_prompt(
        self,
        transcript: str,
        clips: List[Dict],
        duration: float
    ) -> List[Dict]:
        """Build simple multimodal prompt asking for timestamps."""
        
        parts = []
        
        # Instruction
        parts.append({
            "text": f"""You are a video editor. I have a {duration:.1f} second video.

TRANSCRIPT (any language is fine):
"{transcript}"

I have {len(clips)} B-Roll clips to choose from. Here they are:
"""
        })
        
        # Add B-Roll images (1 frame each is sufficient)
        clips_with_images = 0
        for i, clip in enumerate(clips):
            clip_name = clip.get('name', f'clip_{i+1}')
            frames = clip.get("frames", [])
            
            parts.append({"text": f"\n[CLIP {i+1}: {clip_name}]"})
            
            # Add 1 frame (middle frame if multiple)
            if frames:
                frame_path = frames[len(frames) // 2] if len(frames) > 1 else frames[0]
                if Path(frame_path).exists():
                    try:
                        with open(frame_path, "rb") as f:
                            image_data = base64.b64encode(f.read()).decode("utf-8")
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_data
                            }
                        })
                        clips_with_images += 1
                    except:
                        pass
        
        print(f"   🖼️ {clips_with_images}/{len(clips)} clips with images")
        
        # Decision prompt - ask for TIMESTAMPS, MUST USE ALL CLIPS
        parts.append({
            "text": f"""

YOUR TASK:
Create a B-Roll insertion plan for this video.
You MUST use ALL {len(clips)} B-Roll clips - each clip should appear at least once.

OUTPUT FORMAT (JSON only):
{{
  "cuts": [
    {{"start": 2.5, "end": 8.0, "clip": 1, "reason": "brief reason"}},
    {{"start": 10.0, "end": 15.0, "clip": 2, "reason": "brief reason"}},
    {{"start": 18.0, "end": 22.0, "clip": 3, "reason": "brief reason"}}
  ]
}}

RULES:
- start/end = exact seconds (0 to {duration:.1f})
- clip = B-Roll number (1 to {len(clips)})
- ⚠️ MUST USE ALL CLIPS (1 through {len(clips)}) - each at least once!
- Cuts should NOT overlap
- Distribute clips across the video timeline
- Each cut should be 3-8 seconds long

Return ONLY the JSON, nothing else."""
        })
        
        return parts
    
    def _call_vlm(self, parts: List[Dict]) -> str:
        """Call Gemini VLM API."""
        headers = {"Content-Type": "application/json"}
        url = f"{self.endpoint}?key={self.api_key}"
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1000
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        
        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text[:200]}")
        
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    
    def _parse_response(
        self,
        response_text: str,
        available_clips: List[Dict],
        word_timestamps: List[Dict]
    ) -> List[Dict]:
        """Parse VLM response - extract timestamps directly."""
        import re
        
        # Extract JSON
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*"cuts"[^{}]*\[.*?\]\s*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON parse error: {e}")
            print(f"   Response: {response_text[:200]}...")
            return []
        
        cuts = result.get("cuts", [])
        if not cuts:
            print("   ⚠️ No cuts returned by VLM")
            return []
        
        timeline = []
        
        for cut in cuts:
            start = cut.get("start", 0)
            end = cut.get("end", 0)
            clip_num = cut.get("clip", 1)
            reason = cut.get("reason", "")
            
            # Validate
            if end <= start:
                continue
            if not isinstance(clip_num, int) or not (1 <= clip_num <= len(available_clips)):
                continue
            
            clip = available_clips[clip_num - 1]
            
            # Get transcript text for this time range (for display only)
            aroll_text = self._get_text_for_range(start, end, word_timestamps)
            
            timeline.append({
                "aroll_start": start,
                "aroll_end": end,
                "aroll_text": aroll_text,
                "broll_clip": clip.get("path"),
                "broll_id": clip_num - 1,
                "broll_name": clip.get("name", f"clip_{clip_num}"),
                "similarity": 1.0,
                "duration": end - start,
                "llm_reason": reason
            })
        
        # Sort by start time
        timeline = sorted(timeline, key=lambda x: x["aroll_start"])
        
        # VALIDATION: Ensure ALL clips are used
        if word_timestamps and available_clips:
            timeline = self._ensure_all_clips_used(
                timeline, available_clips, word_timestamps
            )
        
        return timeline
    
    def _ensure_all_clips_used(
        self,
        timeline: List[Dict],
        available_clips: List[Dict],
        word_timestamps: List[Dict]
    ) -> List[Dict]:
        """Check if all clips are used, add missing ones."""
        used_clips = set(seg.get("broll_id") for seg in timeline)
        all_clips = set(range(len(available_clips)))
        missing_clips = all_clips - used_clips
        
        if not missing_clips:
            return timeline  # All clips used!
        
        print(f"   ⚠️ {len(missing_clips)} clips not used, adding them...")
        
        # Get video duration
        video_duration = word_timestamps[-1]["end"] if word_timestamps else 30.0
        
        # Find gaps in timeline to insert missing clips
        timeline = sorted(timeline, key=lambda x: x["aroll_start"])
        
        for clip_id in missing_clips:
            clip = available_clips[clip_id]
            
            # Find a gap or extend at end
            insert_time = self._find_insert_position(timeline, video_duration)
            
            if insert_time is not None:
                start, end = insert_time
                aroll_text = self._get_text_for_range(start, end, word_timestamps)
                
                timeline.append({
                    "aroll_start": start,
                    "aroll_end": end,
                    "aroll_text": aroll_text,
                    "broll_clip": clip.get("path"),
                    "broll_id": clip_id,
                    "broll_name": clip.get("name", f"clip_{clip_id+1}"),
                    "similarity": 0.8,
                    "duration": end - start,
                    "llm_reason": "Auto-added (all clips must be used)"
                })
                print(f"      + Added {clip.get('name')} at {start:.1f}s-{end:.1f}s")
        
        return sorted(timeline, key=lambda x: x["aroll_start"])
    
    def _find_insert_position(
        self,
        timeline: List[Dict],
        video_duration: float
    ) -> Optional[tuple]:
        """Find a gap in timeline to insert a clip (4-6 seconds)."""
        target_duration = 5.0  # Target 5 seconds per clip
        
        if not timeline:
            # No existing cuts, use middle of video
            start = video_duration * 0.3
            return (start, min(start + target_duration, video_duration))
        
        # Check gap at beginning
        if timeline[0]["aroll_start"] > target_duration:
            return (0, target_duration)
        
        # Check gaps between segments
        for i in range(len(timeline) - 1):
            gap_start = timeline[i]["aroll_end"]
            gap_end = timeline[i + 1]["aroll_start"]
            gap = gap_end - gap_start
            
            if gap >= target_duration:
                return (gap_start, gap_start + target_duration)
        
        # Check gap at end
        last_end = timeline[-1]["aroll_end"]
        if video_duration - last_end >= target_duration:
            return (last_end, last_end + target_duration)
        
        # No good gap found, just add at end
        return (last_end, min(last_end + target_duration, video_duration))
    
    def _get_text_for_range(
        self,
        start: float,
        end: float,
        word_timestamps: List[Dict]
    ) -> str:
        """Get transcript words that fall within a time range."""
        words = []
        for w in word_timestamps:
            word_start = w.get("start", 0)
            word_end = w.get("end", 0)
            # Word overlaps with range
            if word_end > start and word_start < end:
                words.append(w.get("word", ""))
        
        text = " ".join(words)
        return text[:50] + "..." if len(text) > 50 else text
