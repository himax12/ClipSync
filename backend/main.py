"""
FastAPI Backend - Main Application
Semantic A-Roll/B-Roll Matching Engine
"""

from dotenv import load_dotenv
load_dotenv()  # This reads .env and sets environment variables

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid
import os
from typing import Dict, List
import time
import numpy as np

from backend.core import (
    VisionSensor,
    MemoryLayer,
    SemanticSolver,
    VideoActuator
)
from backend.models import JobStatus, IndexStats

# Initialize FastAPI app
app =FastAPI(
    title="Semantic A-Roll/B-Roll Engine",
    description="AI-powered video matching using Vertex AI (Vision + Speech-to-Text) + FAISS",
    version="0.1.0"
)

# CORS middleware - SECURITY FIX: No more wildcard!
# First Principles: allow_origins=["*"] lets ANY website call your API
# Attackers could trigger video processing using victim's browser session

# Get allowed origins from environment (comma-separated) or use safe defaults
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000,http://127.0.0.1:8501,http://127.0.0.1:3000").split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS if origin.strip()]

print(f"🔒 CORS allowed origins: {CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Restricted to known origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Only needed methods
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],  # Only needed headers
)

# Global state
jobs: Dict[str, Dict] = {}
memory_layer: MemoryLayer = None
vision_sensor: VisionSensor = None
audio_sensor = None  # AudioSensor imported dynamically in startup()

# Configuration from environment
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "firstproject-c5ac2")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))


@app.on_event("startup")
async def startup():
    """
    Initialize models and verify all dependencies.
    
    First Principles:
    - Fail fast: Better to crash on startup than mid-job
    - Clear errors: Tell user exactly what's missing
    - Check everything: FFmpeg, GCP, API keys, disk space
    """
    global memory_layer, vision_sensor, audio_sensor
    
    print("=" * 60)
    print("🚀 Starting ClipSync - AI-Powered B-Roll Insertion")
    print("=" * 60)
    
    # =========================================================================
    # DEPENDENCY HEALTH CHECKS
    # =========================================================================
    print("\n🔍 Running dependency health checks...")
    
    import subprocess
    import shutil
    
    health_issues = []
    
    # 1. Check FFmpeg
    print("   Checking FFmpeg...", end=" ")
    if shutil.which("ffmpeg"):
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
            version = result.stdout.split('\n')[0] if result.stdout else "unknown"
            print(f"✓ ({version[:40]})")
        except Exception as e:
            print(f"✗")
            health_issues.append(f"FFmpeg found but not working: {e}")
    else:
        print("✗")
        health_issues.append("FFmpeg not found. Install with: winget install ffmpeg (Windows) or brew install ffmpeg (Mac)")
    
    # 2. Check FFprobe
    print("   Checking FFprobe...", end=" ")
    if shutil.which("ffprobe"):
        print("✓")
    else:
        print("✗")
        health_issues.append("FFprobe not found. It should come with FFmpeg.")
    
    # 3. Check Gemini API Key
    print("   Checking Gemini API key...", end=" ")
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key and len(gemini_key) > 20:
        print(f"✓ ({gemini_key[:8]}...)")
    else:
        print("✗")
        health_issues.append("GOOGLE_API_KEY not set or invalid. Add to .env file.")
    
    # 4. Check GCP Project ID
    print("   Checking GCP Project ID...", end=" ")
    if PROJECT_ID and PROJECT_ID != "your-project-id":
        print(f"✓ ({PROJECT_ID})")
    else:
        print("⚠️ (using default)")
    
    # 5. Check disk space
    print("   Checking disk space...", end=" ")
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024 ** 3)
        if free_gb > 5:
            print(f"✓ ({free_gb:.1f} GB free)")
        elif free_gb > 1:
            print(f"⚠️ ({free_gb:.1f} GB free - low)")
        else:
            print(f"✗ ({free_gb:.1f} GB free)")
            health_issues.append(f"Low disk space: {free_gb:.1f} GB. Need at least 5GB for video processing.")
    except Exception as e:
        print(f"⚠️ (couldn't check: {e})")
    
    # Report health check results
    if health_issues:
        print("\n" + "=" * 60)
        print("⚠️  HEALTH CHECK WARNINGS:")
        for issue in health_issues:
            print(f"   • {issue}")
        print("=" * 60)
        # Don't crash, just warn - some issues are non-fatal
    else:
        print("   ✅ All dependency checks passed!")
    
    # =========================================================================
    # INITIALIZE COMPONENTS
    # =========================================================================
    
    # Create directories
    for dir_path in [UPLOAD_DIR, OUTPUT_DIR, DATA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize Vision Sensor (Vertex AI)
    print("\n📹 Initializing Vision Sensor...")
    vision_sensor = VisionSensor(project_id=PROJECT_ID, location=LOCATION)
    
    # Initialize Memory Layer (FAISS)
    print("\n🧠 Initializing Memory Layer...")
    memory_layer = MemoryLayer(dimension=1408, use_gpu=True)
    
    # Try to load existing index
    index_path = DATA_DIR / "broll_index"
    try:
        memory_layer.load(str(index_path))
        print(f"✓ Loaded existing index: {memory_layer.get_stats()}")
    except Exception as e:
        print(f"No existing index found (this is normal on first run)")
    
    print("\n🎤 Initializing Audio Sensor (Vertex AI Speech-to-Text)...")
    from backend.core.audio_sensor_vertex import AudioSensor
    audio_sensor = AudioSensor(project_id=PROJECT_ID, location=LOCATION)
    
    print("\n✅ All systems ready!")
    print("=" * 60)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "app": "Semantic A-Roll/B-Roll Engine",
        "version": "0.1.0"
    }


@app.get("/api/index/stats", response_model=IndexStats)
async def get_index_stats():
    """Get current index statistics"""
    if memory_layer is None:
        raise HTTPException(status_code=503, detail="Memory layer not initialized")
    
    return memory_layer.get_stats()


@app.post("/api/upload/broll")
async def upload_broll(files: List[UploadFile] = File(...)):
    """
    Upload and index B-Roll library
    
    This endpoint:
    1. Saves uploaded B-Roll videos
    2. Extracts keyframes
    3. Generates Vertex AI embeddings
    4. Adds to FAISS index
    """
    if memory_layer is None or vision_sensor is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    job_id = str(uuid.uuid4())
    broll_dir = UPLOAD_DIR / "broll" / job_id
    broll_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📤 Uploading {len(files)} B-Roll clips...")
    
    # Save files
    for file in files:
        path = broll_dir / file.filename
        with open(path, "wb") as f:
            content = await file.read()
            f.write(content)
        print(f"  Saved: {file.filename}")
    
    # Process library
    print("\n🔄 Processing B-Roll library...")
    start_time = time.time()
    
    library = vision_sensor.process_broll_library(
        broll_dir,
        fps=float(os.getenv("KEYFRAME_FPS", "0.5"))
    )
    
    # Add to index
    all_embeddings = []
    all_metadata = []
    
    for clip_name, data in library.items():
        # Average embeddings for each clip (simple approach)
        avg_embedding = data["embeddings"].mean(axis=0, keepdims=True)
        all_embeddings.append(avg_embedding)
        all_metadata.append({
            "name": clip_name,
            "path": str(data["path"]),
            "duration": data["duration"]
        })
    
    if all_embeddings:
        memory_layer.add_broll_library(
            np.vstack(all_embeddings),
            all_metadata
        )
        
        # Save index
        index_path = DATA_DIR / "broll_index"
        memory_layer.save(str(index_path))
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ B-Roll indexing complete in {elapsed:.1f}s")
    
    return {
        "status": "success",
        "clips_indexed": len(library),
        "processing_time": elapsed,
        "index_stats": memory_layer.get_stats()
    }


@app.post("/api/process")
async def process_video(
    background_tasks: BackgroundTasks,
    aroll: UploadFile = File(...)
):
    """
    Process A-Roll video with B-Roll matching
    
    This endpoint:
    1. Saves A-Roll video
    2. Starts background processing job
    3. Returns job_id for status tracking
    """
    if any(x is None for x in [memory_layer, vision_sensor, audio_sensor]):
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if memory_layer.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No B-Roll clips indexed. Upload B-Roll library first."
        )
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Waiting to start..."
    }
    
    # Save A-Roll
    aroll_path = UPLOAD_DIR / "aroll" / f"{job_id}.mp4"
    aroll_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(aroll_path, "wb") as f:
        content = await aroll.read()
        f.write(content)
    
    print(f"\n🎬 New processing job: {job_id}")
    
    # Background processing
    background_tasks.add_task(
        process_pipeline,
        job_id,
        aroll_path
    )
    
    return {"job_id": job_id, "status": "queued"}

@app.post("/api/process/json")
async def process_from_json(
    background_tasks: BackgroundTasks,
    json_data: dict
):
    """
    Process videos from JSON input with URLs
    
    Expects JSON format:
    {
      "a_roll": {"url": "https://...", "metadata": "..."},
      "b_rolls": [
        {"id": "broll_1", "url": "https://...", "metadata": "..."},
        ...
      ]
    }
    """
    if any(x is None for x in [memory_layer, vision_sensor, audio_sensor]):
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Extract B-Roll URLs
    b_rolls = json_data.get("b_rolls", [])
    if not b_rolls:
        raise HTTPException(status_code=400, detail="No B-Roll videos found in JSON")
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "downloading",
        "progress": 0,
        "message": "Downloading videos from URLs..."
    }
    
    print(f"\n🌐 New JSON processing job: {job_id}")
    print(f"   B-Roll clips to download: {len(b_rolls)}")
    
    # Background processing
    background_tasks.add_task(
        process_json_pipeline,
        job_id,
        json_data
    )
    
    return {"job_id": job_id, "status": "queued"}


async def process_json_pipeline(job_id: str, json_data: dict):
    """Background processing for JSON input"""
    from backend.core.downloader import download_video, download_all_videos_async
    import asyncio
    
    try:
        # Create download directory
        download_dir = UPLOAD_DIR / "json" / job_id
        download_dir.mkdir(parents= True, exist_ok=True)
        
        # Download B-Roll videos IN PARALLEL (First Principles: I/O bound = async wins!)
        # Old: Sequential → N videos × 10s = 60s
        # New: Parallel → N videos in ~10s (slowest download time)
        jobs[job_id]["message"] = "Downloading B-Roll videos in parallel..."
        jobs[job_id]["progress"] = 5
        
        b_rolls = json_data.get("b_rolls", [])
        
        # Build list of (url, output_path) tuples for parallel download
        download_tasks = []
        for i, broll in enumerate(b_rolls):
            url = broll.get("url")
            if not url:
                continue
            broll_id = broll.get("id", f"broll_{i}")
            output_path = download_dir / f"{broll_id}.mp4"
            download_tasks.append((url, output_path))
        
        # Execute all downloads in parallel!
        results = await download_all_videos_async(download_tasks)
        
        # Collect successful downloads
        broll_files = [
            path for (url, path), success in zip(download_tasks, results) if success
        ]
        
        jobs[job_id]["progress"] = 20
        
        if not broll_files:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Failed to download any B-Roll videos"
            return
        
        # Process B-Roll library
        jobs[job_id]["message"] = "Processing B-Roll library..."
        jobs[job_id]["progress"] = 25
        
        
        # Process B-Roll library (HARDCODED 1.0 FPS for short clips)
        # NOTE: Don't use KEYFRAME_FPS from .env - it's set too low (0.1)
        library = vision_sensor.process_broll_library(
            download_dir,
            fps=1.0  # Must be >= 0.25 for 4-second clips to extract frames!
        )
        
        
        # Try to use Gemini Vision for auto-generated metadata
        # Falls back to JSON metadata if GOOGLE_API_KEY not available
        use_gemini = False
        describer = None
        
        try:
            from backend.core.frame_describer import FrameDescriber
            describer = FrameDescriber()
            use_gemini = True
            print("  🤖 Using Gemini Vision for automatic metadata generation")
        except Exception as e:
            print(f"  ⚠️  Gemini Vision not available (add GOOGLE_API_KEY to .env): {e}")
            print("  📝 Falling back to JSON metadata")
        
        # Add to FAISS index
        all_embeddings = []
        all_metadata = []
        
        for i, (clip_name, data) in enumerate(library.items()):
            # Get frame paths for VLM analysis (the VLM will SEE these!)
            frame_paths = data.get("frames", [])
            
            # Get JSON metadata if available (fallback for FAISS embedding only)
            b_roll_meta = json_data["b_rolls"][i] if i < len(json_data.get("b_rolls", [])) else {}
            json_metadata = b_roll_meta.get("metadata", clip_name)
            
            if frame_paths:
                print(f"  🖼️ {clip_name}: {len(frame_paths)} frames ready for VLM analysis")
            else:
                print(f"  📝 {clip_name}: No frames, using metadata: {json_metadata[:40]}...")
            
            # Embed metadata text for FAISS (used as backup only)
            text_embedding = vision_sensor.embed_text(json_metadata)
            
            all_embeddings.append(text_embedding.reshape(1, -1))
            all_metadata.append({
                "name": clip_name,
                "path": str(data["path"]),
                "duration": data["duration"],
                "metadata": json_metadata,
                "frames": frame_paths  # VLM will SEE these!
            })
        
        if all_embeddings:
            # Clear old index to prevent stale paths from previous jobs
            job_memory_layer = MemoryLayer(dimension=1408, use_gpu=False)
            
            job_memory_layer.add_broll_library(
                np.vstack(all_embeddings),
                all_metadata
            )
            
            # Save index (for this job only)
            index_path = DATA_DIR / f"broll_index_{job_id}"
            job_memory_layer.save(str(index_path))
        else:
            # NO B-Roll clips processed successfully
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Failed to process any B-Roll clips. Check that videos are valid and FFmpeg is working."
            print(f"[{job_id}] ❌ No B-Roll clips processed successfully")
            return
        
        jobs[job_id]["progress"] = 40
        
        # Download and process A-Roll if provided
        a_roll = json_data.get("a_roll")
        if a_roll and a_roll.get("url"):
            jobs[job_id]["message"] = "Downloading A-Roll video..."
            jobs[job_id]["progress"] = 45
            
            aroll_path = download_dir / "aroll.mp4"
            
            if not download_video(a_roll["url"], aroll_path):
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"] = "Failed to download A-Roll video"
                return
            
            # Now process like normal
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["progress"] = 50
            
            # Extract audio (wrapped in thread - subprocess is I/O bound)
            # First Principles: subprocess.run() blocks the event loop
            # asyncio.to_thread() runs it in a thread pool, freeing the event loop
            jobs[job_id]["message"] = "Extracting audio..."
            audio_path = aroll_path.with_suffix(".wav")
            await asyncio.to_thread(audio_sensor.extract_audio, aroll_path, audio_path)
            jobs[job_id]["progress"] = 55
            
            # Transcribe (wrapped in thread - network I/O to Vertex AI)
            jobs[job_id]["message"] = "Transcribing with Vertex AI Speech-to-Text..."
            aligned = await asyncio.to_thread(audio_sensor.transcribe_and_align, audio_path)
            jobs[job_id]["progress"] = 70
            
            # AUTONOMOUS LLM EDITOR: One call does EVERYTHING
            jobs[job_id]["message"] = "🧠 LLM making all editing decisions..."
            
            # Build transcript
            word_timestamps = aligned.get("word_segments", [])
            full_transcript = " ".join([w["word"] for w in word_timestamps])
            
            if not full_transcript.strip():
                # Fallback to segment text
                full_transcript = " ".join([s["text"] for s in aligned.get("segments", [])])
            
            # Use Autonomous Editor via dependency injection
            # VLM call wrapped in thread - network I/O to Gemini API
            from backend.core import get_container
            container = get_container()
            editor = container.get_editor()
            timeline = await asyncio.to_thread(
                editor.create_timeline,
                full_transcript,
                word_timestamps,
                all_metadata  # B-Roll options
            )
            
            jobs[job_id]["progress"] = 85
            
            # Calculate stats
            stats = {
                "total_segments": len(timeline),
                "matched_segments": sum(1 for t in timeline if t.get("broll_clip")),
                "match_rate": 1.0 if timeline else 0.0,
                "avg_similarity": 1.0,  # LLM selection, not vector
            }
            jobs[job_id]["progress"] = 85
            
            # Assemble video (wrapped in thread - FFmpeg subprocess is I/O bound)
            jobs[job_id]["message"] = "Assembling final video..."
            output_path = OUTPUT_DIR / f"{job_id}_final.mp4"
            await asyncio.to_thread(
                VideoActuator.assemble_timeline, aroll_path, timeline, output_path
            )
            
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = "Processing complete!"
            jobs[job_id]["output_path"] = str(output_path)
            jobs[job_id]["stats"] = stats
            
            print(f"[{job_id}] ✅ Complete!")
        else:
            # Just B-Roll indexing
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = "B-Roll indexing complete! No A-Roll provided."
            print(f"[{job_id}] ✅ B-Roll indexed!")
            
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Error: {str(e)}"
        print(f"[{job_id}] ❌ Error: {e}")


async def process_pipeline(job_id: str, aroll_path: Path):
    """Background processing pipeline"""
    try:
        jobs[job_id]["status"] = "processing"
        
        # Step 1: Extract audio
        jobs[job_id]["message"] = "Extracting audio..."
        jobs[job_id]["progress"] = 10
        print(f"[{job_id}] Extracting audio...")
        
        audio_path = aroll_path.with_suffix(".wav")
        audio_sensor.extract_audio(aroll_path, audio_path)
        
        # Step 2: Transcribe & align
        jobs[job_id]["message"] = "Transcribing with WhisperX..."
        jobs[job_id]["progress"] = 20
        print(f"[{job_id}] Transcribing...")
        
        aligned = audio_sensor.transcribe_and_align(audio_path)
        
        # Step 3: Create segments
        jobs[job_id]["message"] = "Creating semantic segments..."
        jobs[job_id]["progress"] = 50
        print(f"[{job_id}] Creating segments...")
        
        segments = audio_sensor.create_semantic_segments(
            aligned,
            min_duration=float(os.getenv("MIN_SEGMENT_DURATION", "3.0")),
            max_duration=float(os.getenv("MAX_SEGMENT_DURATION", "10.0"))
        )
        
        # Step 4: Match with B-Roll
        jobs[job_id]["message"] = "Matching with B-Roll..."
        jobs[job_id]["progress"] = 60
        print(f"[{job_id}] Matching...")
        
        solver = SemanticSolver(memory_layer, vision_sensor, PROJECT_ID)
        timeline = solver.solve(segments, k_candidates=5, min_similarity=0.08, allow_reuse=True)  # Allow reuse since clips are short
        stats = solver.get_match_stats(timeline)
        
        # Step 5: Assemble video
        jobs[job_id]["message"] = "Assembling final video..."
        jobs[job_id]["progress"] = 80
        print(f"[{job_id}] Assembling video...")
        
        output_path = OUTPUT_DIR / f"{job_id}_final.mp4"
        VideoActuator.assemble_timeline(aroll_path, timeline, output_path)
        
        # Complete
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = "Processing complete!"
        jobs[job_id]["output_path"] = str(output_path)
        jobs[job_id]["stats"] = stats
        
        print(f"[{job_id}] ✅ Complete!")
        
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Error: {str(e)}"
        print(f"[{job_id}] ❌ Error: {e}")


@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Get processing job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """Download processed video"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "complete":
        raise HTTPException(status_code=400, detail="Job not complete")
    
    output_path = Path(job["output_path"])
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"semantic_match_{job_id}.mp4"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
