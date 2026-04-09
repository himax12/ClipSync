"""
Topic Identifier Module
Identifies topics in transcript segments using Gemini LLM
"""

import os
import json
import re
import requests
from typing import Dict, List


class TopicIdentifier:
    def __init__(self, api_key: str, model: str = "gemini-3.1-pro-preview"):
        self.api_key = api_key
        self.model = model

    def identify_topics(self, transcript_text: str) -> List[str]:
        """Identify topics present in transcript"""
        if not transcript_text or not transcript_text.strip():
            return []

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json"}

        prompt = f"""Analyze this transcript and identify 3-5 topics present.

TRANSCRIPT: "{transcript_text[:2000]}"

TASK: Extract the main topics covered in this transcript.
Topics should be simple labels like: cooking, travel, technology, business, music, sports, etc.

OUTPUT FORMAT (JSON only):
{{"topics": ["topic1", "topic2", "topic3"]}}

RULES:
- Only return topics that are clearly discussed
- Use simple, single-word or short-phrase topic names
- Be specific but not overly niche"""

        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 100}
            }

            response = requests.post(
                f"{endpoint}?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                return []

            result_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]

            json_match = re.search(r'\{"topics"\s*:\s*\[.*?\]\s*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(result_text)

            return result.get("topics", [])

        except Exception:
            return []

    def tag_segments(
        self,
        segments: List[Dict],
        full_transcript: str
    ) -> List[Dict]:
        """Tag each segment with relevant topics"""
        if not segments:
            return segments

        global_topics = self.identify_topics(full_transcript)

        tagged_segments = []
        for segment in segments:
            segment_topics = self._get_segment_topics(
                segment.get("text", ""),
                global_topics
            )
            tagged_segments.append({
                **segment,
                "topics": segment_topics
            })

        return tagged_segments

    def _get_segment_topics(
        self,
        segment_text: str,
        global_topics: List[str]
    ) -> List[str]:
        """Determine which topics a segment relates to"""
        if not segment_text or not global_topics:
            return []

        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json"}

        topics_str = ", ".join(global_topics[:5])
        prompt = f"""Given these global topics: [{topics_str}]
And this segment: "{segment_text[:500]}"

Identify which 1-2 topics from the global list this segment relates to.
If none clearly apply, return empty array.

OUTPUT FORMAT (JSON only):
{{"segment_topics": ["topic1"]}}"""

        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 50}
            }

            response = requests.post(
                f"{endpoint}?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=15
            )

            if response.status_code != 200:
                return []

            result_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]

            json_match = re.search(
                r'\{"segment_topics"\s*:\s*\[.*?\]\s*\}',
                result_text,
                re.DOTALL
            )
            if json_match:
                result = json.loads(json_match.group())
                return result.get("segment_topics", [])

        except Exception:
            pass

        return []


def create_topic_identifier(api_key: str = None) -> TopicIdentifier:
    """Factory function with dependency injection"""
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY", "")
    return TopicIdentifier(api_key=api_key)
