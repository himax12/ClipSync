"""
LLM-as-Judge Module
Uses Gemini to reason about which B-Roll best supports narrative context.

First Principles Design:
1. Vector similarity finds CANDIDATES (fast, approximate)
2. LLM reasons about NARRATIVE FIT (slow, accurate)
3. Combine both for optimal quality/speed tradeoff
"""

import os
import json
import requests
import base64
from typing import List, Dict, Optional
from pathlib import Path


class LLMJudge:
    """
    Uses Gemini to select B-Roll based on narrative reasoning,
    not just vector similarity.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM Judge with Gemini API.
        
        Args:
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # Prompt template with Chain-of-Thought reasoning
        self.prompt_template = """You are a professional video editor selecting B-Roll clips.

TASK: Select the B-Roll that BEST supports the narrative moment.

NARRATIVE SEGMENT:
"{segment_text}"

B-ROLL CANDIDATES:
{candidates_text}

EVALUATION CRITERIA:
1. Narrative Support: Does this visual reinforce what's being said?
2. Emotional Alignment: Does the mood match the speaker's tone?
3. Contextual Fit: Is it appropriate for the subject matter?
4. Visual Interest: Will it engage the viewer?

THINK STEP-BY-STEP:
1. What is the speaker's core message?
2. What visual would best reinforce this message?
3. Which candidate aligns with the emotional tone?
4. Rank all candidates by narrative fit.

OUTPUT FORMAT (JSON only, no other text):
{{
  "best": <candidate_number>,
  "scores": [<score_1>, <score_2>, ...],
  "reason": "<one sentence explaining why the best choice supports the narrative>"
}}

IMPORTANT:
- Scores are 1-10 (10 = perfect narrative support)
- "best" is the candidate NUMBER (1-indexed)
- Choose based on STORY FIT, not visual complexity"""

    def rank_candidates(
        self,
        segment_text: str,
        candidates: List[Dict],
        include_frames: bool = False
    ) -> Dict:
        """
        Use LLM to rank B-Roll candidates by narrative fit.
        
        Args:
            segment_text: The A-Roll narrative text
            candidates: List of B-Roll candidates with descriptions
            include_frames: Whether to include frame images (multimodal)
        
        Returns:
            Dict with 'best' (index), 'scores' (list), 'reason' (str)
        """
        # Build candidates text
        candidates_text = "\n".join([
            f"{i+1}. {c.get('name', f'clip_{i}')}: {c.get('description', 'No description')}"
            for i, c in enumerate(candidates)
        ])
        
        # Format prompt
        prompt = self.prompt_template.format(
            segment_text=segment_text,
            candidates_text=candidates_text
        )
        
        # Call Gemini API
        try:
            result = self._call_gemini(prompt)
            
            # Parse JSON response
            parsed = self._parse_response(result, len(candidates))
            
            print(f"  🧠 LLM Judge: Best={parsed['best']} | Reason: {parsed['reason'][:60]}...")
            
            return parsed
            
        except Exception as e:
            print(f"  ⚠️ LLM Judge failed: {e}")
            # Fallback: return first candidate
            return {
                "best": 1,
                "scores": [5] * len(candidates),
                "reason": "Fallback selection (LLM error)"
            }
    
    def _call_gemini(self, prompt: str) -> str:
        """Make REST API call to Gemini"""
        
        headers = {"Content-Type": "application/json"}
        url = f"{self.endpoint}?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.2,  # Low for consistency
                "maxOutputTokens": 200,
                "topP": 0.8
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API error: {response.status_code} - {response.text[:200]}")
        
        data = response.json()
        
        # Extract text from response
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text.strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemini response structure: {e}")
    
    def _parse_response(self, response_text: str, num_candidates: int) -> Dict:
        """Parse LLM response to extract ranking"""
        
        # Try to extract JSON from response
        # Handle cases where LLM adds extra text around JSON
        
        try:
            # Try direct JSON parse first
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON block in text
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    result = None
            else:
                result = None
        
        if not result:
            # Fallback if parsing fails
            return {
                "best": 1,
                "scores": [5] * num_candidates,
                "reason": "Could not parse LLM response"
            }
        
        # Validate and normalize
        best = result.get("best", 1)
        if isinstance(best, int) and 1 <= best <= num_candidates:
            pass
        else:
            best = 1
        
        scores = result.get("scores", [5] * num_candidates)
        if not isinstance(scores, list) or len(scores) != num_candidates:
            scores = [5] * num_candidates
        
        reason = result.get("reason", "No reason provided")
        if not isinstance(reason, str):
            reason = str(reason)
        
        return {
            "best": best,
            "scores": scores,
            "reason": reason
        }

    def rank_with_frames(
        self,
        segment_text: str,
        candidates: List[Dict],
        frame_paths: List[Path]
    ) -> Dict:
        """
        Multimodal ranking using actual video frames.
        
        This is more expensive but more accurate since the LLM
        can SEE the actual visuals, not just descriptions.
        """
        # Build multimodal prompt with images
        parts = [{"text": f"NARRATIVE: {segment_text}\n\nB-ROLL OPTIONS:\n"}]
        
        for i, (candidate, frame_path) in enumerate(zip(candidates, frame_paths)):
            # Add text label
            parts.append({"text": f"\n{i+1}. {candidate.get('name', f'clip_{i}')}:"})
            
            # Add image if available
            if frame_path and Path(frame_path).exists():
                with open(frame_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_data
                    }
                })
        
        # Add instruction
        parts.append({
            "text": """
            
Based on the NARRATIVE and the B-ROLL images shown:
Which B-Roll BEST supports this narrative moment?

OUTPUT (JSON only):
{"best": <number>, "scores": [<score_1>, ...], "reason": "<why>"}"""
        })
        
        # Call Gemini with multimodal content
        headers = {"Content-Type": "application/json"}
        url = f"{self.endpoint}?key={self.api_key}"
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 200
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                raise RuntimeError(f"Gemini API error: {response.status_code}")
            
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return self._parse_response(text.strip(), len(candidates))
            
        except Exception as e:
            print(f"  ⚠️ Multimodal ranking failed: {e}")
            # Fallback to text-only
            return self.rank_candidates(segment_text, candidates)
