"""
Semantic Solver Module
Greedy matching algorithm with lookahead and constraint validation
"""

import os
import numpy as np
from typing import List, Dict, Set
from .memory_layer import MemoryLayer
from .vision_sensor import VisionSensor


class SemanticSolver:
    """
    LLM-as-Judge Semantic Solver
    
    Uses Gemini to reason about narrative fit, not just vector similarity.
    This is the ONLY matching strategy - no fallbacks.
    """
    
    def __init__(self, memory: MemoryLayer, vision_sensor: VisionSensor, project_id: str):
        """Initialize solver with LLM Judge as primary matcher"""
        self.memory = memory
        self.vision_sensor = vision_sensor
        
        # Translator for multilingual support
        from backend.core.translator import TextTranslator
        self.translator = TextTranslator(project_id=project_id)
        
        # LLM Judge is REQUIRED - no fallback
        from backend.core.llm_judge import LLMJudge
        self.llm_judge = LLMJudge()
        print("✓ LLM-as-Judge initialized (Primary Matching Engine)")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text for FAISS candidate retrieval"""
        return self.vision_sensor.embed_text(text)
    
    def solve(
        self,
        aroll_segments: List[Dict],
        k_candidates: int = 5,
        allow_reuse: bool = True,
        min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Pure LLM-as-Judge Matching
        
        Pipeline:
        1. FAISS: Get top-K candidates (fast retrieval)
        2. LLM: Reason about narrative fit (smart selection)
        
        This is the ONLY strategy - we trust LLM for all decisions.
        """
        print(f"\n🧠 LLM-as-Judge Matching ({len(aroll_segments)} segments)")
        print("━" * 50)
        
        timeline_plan = []
        used_clips: Set[int] = set()
        translated_segments = self.translator.translate_segments(aroll_segments)
        
        for i, segment in enumerate(translated_segments):
            print(f"\n📝 Segment {i+1}/{len(translated_segments)}")
            print(f"   \"{segment['text'][:60]}...\"")
            
            # Stage 1: FAISS Candidate Retrieval
            text_vector = self.encode_text(segment["text"])
            similarities, indices = self.memory.search(text_vector, k=k_candidates)
            
            # Build candidate list
            candidates = []
            for j, idx in enumerate(indices[0]):
                if idx == -1:
                    continue
                if not allow_reuse and idx in used_clips:
                    continue
                    
                clip_meta = self.memory.clip_metadata[idx]
                sim = float(similarities[0][j])
                
                candidates.append({
                    "idx": idx,
                    "name": clip_meta["name"],
                    "description": clip_meta.get("metadata", clip_meta.get("description", "No description")),
                    "path": clip_meta["path"],
                    "similarity": sim,
                    "duration": clip_meta.get("duration", 0)
                })
            
            if not candidates:
                timeline_plan.append({
                    "aroll_start": segment["start"],
                    "aroll_end": segment["end"],
                    "aroll_text": segment["text"],
                    "broll_clip": None,
                    "broll_name": None,
                    "similarity": 0.0,
                    "llm_score": 0,
                    "llm_reason": "No candidates available",
                    "duration": segment["duration"]
                })
                print(f"   ⚠️ No candidates found")
                continue
            
            # Show FAISS candidates
            print(f"   📊 FAISS candidates ({len(candidates)}):")
            for c in candidates:
                print(f"      • {c['name']}: {c['description'][:40]}... (sim={c['similarity']:.3f})")
            
            # Stage 2: LLM Narrative Reasoning
            llm_result = self.llm_judge.rank_candidates(
                segment_text=segment["text"],
                candidates=candidates
            )
            
            # Get LLM's choice
            best_idx = llm_result["best"] - 1
            if 0 <= best_idx < len(candidates):
                best = candidates[best_idx]
            else:
                best = candidates[0]
            
            # Record result
            timeline_plan.append({
                "aroll_start": segment["start"],
                "aroll_end": segment["end"],
                "aroll_text": segment["text"],
                "broll_clip": best["path"],
                "broll_id": best["idx"],
                "broll_name": best["name"],
                "similarity": best["similarity"],
                "llm_score": llm_result["scores"][best_idx] if best_idx < len(llm_result["scores"]) else 5,
                "llm_reason": llm_result["reason"],
                "duration": segment["duration"]
            })
            
            used_clips.add(best["idx"])
            
            # Show LLM decision
            print(f"   🎯 LLM Selected: {best['name']}")
            print(f"      Reason: {llm_result['reason']}")
        
        # Summary
        matched = sum(1 for e in timeline_plan if e["broll_clip"])
        print(f"\n{'━' * 50}")
        print(f"✅ Matched {matched}/{len(aroll_segments)} segments")
        
        return timeline_plan
    
    def get_match_stats(self, timeline_plan: List[Dict]) -> Dict:
        """Get statistics including LLM scores"""
        matched = [e for e in timeline_plan if e["broll_clip"]]
        
        if not matched:
            return {
                "total_segments": len(timeline_plan),
                "matched_segments": 0,
                "match_rate": 0.0,
                "avg_similarity": 0.0,
                "avg_llm_score": 0.0
            }
        
        similarities = [e["similarity"] for e in matched]
        llm_scores = [e.get("llm_score", 5) for e in matched]
        
        return {
            "total_segments": len(timeline_plan),
            "matched_segments": len(matched),
            "match_rate": len(matched) / len(timeline_plan),
            "avg_similarity": np.mean(similarities),
            "avg_llm_score": np.mean(llm_scores),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities)
        }



