"""
Frame Description Module - Auto-generate metadata from B-Roll frames
Uses Gemini Vision REST API (reliable, not SDK)
"""

from pathlib import Path
from typing import List
import os
import base64
import requests


class FrameDescriber:
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini Vision for frame description
        
        Args:
            api_key: Google AI API key (or use GOOGLE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        print("✓ Gemini Vision initialized for frame description")
    
    def describe_frame(self, image_path: Path) -> str:
        """
        Generate semantic description of a video frame
        
        Args:
            image_path: Path to frame image
        
        Returns:
            Semantic description text
        """
        try:
            # Read and encode image as base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # SEMANTIC ANALYSIS prompt (research-backed, forces conceptual thinking)
            prompt = """You are a semantic video analyzer. Your job is to extract ABSTRACT CONCEPTS from images.

CRITICAL RULES:
1. NEVER describe what you literally see (no "image shows...", no objects, no people)
2. ONLY output thematic concepts the scene represents
3. Think: What does this MEAN for a story/narrative?

ANALYSIS FRAMEWORK:
• Purpose: What human activity or value? (preparation, quality control, social bonding, consumption)
• Emotion: What atmosphere? (casual/formal, energetic/calm, traditional/modern)
• Culture: What social context? (urban lifestyle, street culture, professional environment, home setting)
• Theme: What abstract ideas? (health consciousness, quality standards, community, tradition)

STRICT OUTPUT FORMAT:
- Comma-separated list of 4-6 concepts
- Each concept: 1-3 words maximum
- Abstract concepts ONLY (not "food stall" but "street culture")
- Example: "urban lifestyle, informal dining, food preparation, cultural tradition, social interaction"

Now analyze THIS image and output ONLY the concept list:"""
            
            # Build request payload
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_data
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.3,  # Lower for more consistent conceptual extraction
                    "maxOutputTokens": 80,  # Force concise output
                    "topP": 0.8  # Focus on high-confidence concepts
                }
            }
            
            # Make REST API call
            response = requests.post(
                f"{self.endpoint}?key={self.api_key}",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                description = result['candidates'][0]['content']['parts'][0]['text'].strip()
                return description
            else:
                raise Exception(f"API error {response.status_code}: {response.text}")
        
        except Exception as e:
            print(f"⚠️  Frame description failed: {e}")
            return f"Video frame from {image_path.stem}"  # Fallback
    
    def describe_frames_aggregate(self, frame_paths: List[Path]) -> str:
        """
        Generate aggregate description from multiple frames
        
        Args:
            frame_paths: List of frame paths from same video
        
        Returns:
            Aggregated semantic description
        """
        if not frame_paths:
            return "No frames available"
        
        # For now, describe first and middle frames, then combine
        if len(frame_paths) == 1:
            return self.describe_frame(frame_paths[0])
        
        # Describe first and middle frame
        first_desc = self.describe_frame(frame_paths[0])
        mid_idx = len(frame_paths) // 2
        mid_desc = self.describe_frame(frame_paths[mid_idx])
        
        # Combine intelligently
        if first_desc == mid_desc:
            return first_desc
        else:
            return f"{first_desc} {mid_desc}"
