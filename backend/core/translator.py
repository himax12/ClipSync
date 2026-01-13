"""
Translation Module - Auto-translate non-English text to English
Uses Google Cloud Translation API
"""

from google.cloud import translate_v2 as translate
from typing import List, Dict
import os


class TextTranslator:
    def __init__(self, project_id: str):
        """
        Initialize translator with GCP credentials
        
        Args:
            project_id: GCP project ID
        """
        self.client = translate.Client()
        self.project_id = project_id
        print(f"✓ Cloud Translation initialized")
    
    def translate_to_english(self, text: str, source_language: str = None) -> Dict[str, str]:
        """
        Translate text to English
        
        Args:
            text: Text to translate
            source_language: Source language code (auto-detected if None)
        
        Returns:
            Dict with 'original', 'translated', 'detected_language'
        """
        if not text or not text.strip():
            return {
                "original": text,
                "translated": text,
                "detected_language": "unknown"
            }
        
        try:
            # Translate to English
            result = self.client.translate(
                text,
                target_language='en',
                source_language=source_language
            )
            
            return {
                "original": text,
                "translated": result['translatedText'],
                "detected_language": result.get('detectedSourceLanguage', source_language or 'unknown')
            }
        
        except Exception as e:
            print(f"⚠️  Translation failed: {e}, using original text")
            return {
                "original": text,
                "translated": text,  # Fallback to original
                "detected_language": "error"
            }
    
    def translate_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Translate all segments to English, preserving original
        
        Args:
            segments: List of segments with 'text' field
        
        Returns:
            Same segments with added 'text_en' and 'original_language' fields
        """
        translated_segments = []
        
        for segment in segments:
            translation = self.translate_to_english(segment['text'])
            
            # Add translation while preserving original
            segment_copy = segment.copy()
            segment_copy['text_original'] = translation['original']
            segment_copy['text'] = translation['translated']  # Replace with English
            segment_copy['detected_language'] = translation['detected_language']
            
            translated_segments.append(segment_copy)
        
        # Print summary
        languages = set(s['detected_language'] for s in translated_segments)
        if 'en' not in languages or len(languages) > 1:
            print(f"✓ Translated {len(segments)} segments from {', '.join(languages)} → English")
        
        return translated_segments
