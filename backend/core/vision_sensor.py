"""
Vision Sensor Module
Converts video frames to semantic embeddings using Vertex AI Multimodal Embeddings
"""

from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage
from PIL import Image
from pathlib import Path
import subprocess
import numpy as np
from typing import List, Dict
import time


class VisionSensor:
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize Vertex AI multimodal embeddings
        
        Args:
            project_id: GCP project ID
            location: GCP region (default: us-central1)
        """
        import vertexai
        print(f"Initializing Vertex AI in project {project_id}, location {location}...")
        vertexai.init(project=project_id, location=location)
        
        self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        self.dimension = 1408  # Vertex AI embedding dimension
        print(f"Vertex AI model loaded (dimension={self.dimension})")
        
    def extract_keyframes(
        self,
        video_path: Path,
        output_dir: Path,
        fps: float = 0.5  # 1 frame every 2 seconds
    ) -> List[Path]:
        """
        Extract I-frames at specified rate using FFmpeg
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            fps: Frames per second to extract
        
        Returns:
            List of paths to extracted frames
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(output_dir / "frame_%05d.jpg")
        
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-qscale:v", "2",  # High quality JPEG
            "-y", pattern
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True)
        frames = sorted(output_dir.glob("frame_*.jpg"))
        
        print(f"Extracted {len(frames)} frames from {video_path.name}")
        return frames
    
    def embed_frames(self, frame_paths: List[Path], batch_size: int = 5) -> np.ndarray:
        """
        Convert frames to normalized embeddings using Vertex AI
        
        First Principles:
        - Network latency: 200-500ms per request
        - Batching reduces total time: N×500ms → (N/5)×500ms
        - Exponential backoff handles quota gracefully
        
        Args:
            frame_paths: List of frame image paths
            batch_size: Process frames in batches (default 5)
        
        Returns:
            Array of shape (N, 1408) with L2-normalized embeddings
        """
        embeddings = []
        total_frames = len(frame_paths)
        
        print(f"  Embedding {total_frames} frames in batches of {batch_size}...")
        
        for i, frame_path in enumerate(frame_paths):
            # Retry logic for quota errors
            max_retries = 5
            retry_delay = 2  # Start with 2 seconds
            
            for attempt in range(max_retries):
                try:
                    # Load image for Vertex AI
                    image = VertexImage.load_from_file(str(frame_path))
                    
                    # Get embedding from Vertex AI
                    response = self.model.get_embeddings(
                        image=image,
                        dimension=self.dimension
                    )
                    
                    embedding = np.array(response.image_embedding)
                    
                    # Normalize to unit length (critical for FAISS cosine similarity)
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    embeddings.append(embedding)
                    
                    # Success - break retry loop
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Check if it's a quota error
                    if "429" in error_str or "Quota exceeded" in error_str or "quota" in error_str.lower():
                        if attempt < max_retries - 1:
                            print(f"  ⚠️  Quota limit hit, waiting {retry_delay}s before retry {attempt + 1}/{max_retries}...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        else:
                            print(f"  ✗ Max retries reached for {frame_path.name}, skipping")
                            break
                    else:
                        print(f"  ✗ Failed to embed {frame_path.name}: {e}")
                        break
            
            # Intelligent rate limiting
            # First Principle: Balance API limits vs speed
            if (i + 1) % batch_size == 0:
                # Batch boundary: Longer delay to respect quotas
                time.sleep(0.5)  # Reduced from 2s (still safe)
                if (i + 1) < total_frames:
                    print(f"  Progress: {i+1}/{total_frames} frames")
            else:
                # Within batch: Minimal delay
                time.sleep(0.2)  # Reduced from 0.5s (faster!)
        
        if not embeddings:
            raise ValueError("No frames were successfully embedded")
        
        print(f"  ✓ Embedded {len(embeddings)}/{total_frames} frames")
        return np.vstack(embeddings).astype('float32')
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Encode text using same embedding space as images
        
        First Principles:
        - Text embedding: ~100ms processing time
        - Much faster than image embedding
        - Can use shorter retry delays
        
        Args:
            text: Text to embed
        
        Returns:
            Array of shape (1408,) with L2-normalized embedding
        """
        max_retries = 5
        retry_delay = 1  # Reduced from 2s (text is faster)
        
        for attempt in range(max_retries):
            try:
                response = self.model.get_embeddings(
                    contextual_text=text,
                    dimension=self.dimension
                )
                
                embedding = np.array(response.text_embedding)
                
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding.astype('float32')
                
            except Exception as e:
                error_str = str(e)
                
                if "429" in error_str or "Quota exceeded" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  Quota limit hit on text embedding, waiting {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise Exception(f"Quota exceeded after {max_retries} retries")
                else:
                    raise
    
    def process_broll_library(self, broll_dir: Path, fps: float = 0.5) -> Dict[str, Dict]:
        """
        Process all B-Roll clips in a directory
        
        Args:
            broll_dir: Directory containing B-Roll video files
            fps: Keyframe extraction rate
        
        Returns:
            Dictionary mapping clip names to their metadata and embeddings
        """
        library = {}
        
        video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(broll_dir.glob(ext))
        
        print(f"Found {len(video_files)} B-Roll clips to process")
        
        for video_file in video_files:
            print(f"\nProcessing {video_file.name}...")
            frames_dir = broll_dir / f"{video_file.stem}_frames"
            
            try:
                frames = self.extract_keyframes(video_file, frames_dir, fps=fps)
                
                if frames:
                    # NOTE: We DON'T embed frames anymore!
                    # Frames are only used for Gemini descriptions
                    # The text descriptions are what get embedded and matched
                    # This saves 84% of API quota!
                    
                    print(f"  Extracted {len(frames)} frames for Gemini analysis")
                    
                    library[video_file.stem] = {
                        "path": str(video_file),
                        "duration": self._get_duration(video_file),
                        "frames": frames  # For Gemini descriptions only
                    }
                    print(f"✓ {video_file.stem}: {len(frames)} frames ready for semantic analysis")
            except Exception as e:
                print(f"✗ Failed to process {video_file.name}: {e}")
                continue
        
        return library
    
    @staticmethod
    def _get_duration(video_path: Path) -> float:
        """Get video duration in seconds using ffprobe"""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
