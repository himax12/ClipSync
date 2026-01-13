"""
Test Script for Video JSON Input
Tests the semantic matching engine with a JSON file containing video information
"""

import json
import requests
from pathlib import Path
import time
import sys

# Configuration
API_BASE = "http://localhost:8000/api"
JSON_FILE = r"c:\Users\ghima\Downloads\video_url.json"  # User's JSON file


def load_video_json(json_path: str):
    """Load and parse the video JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def download_video(url: str, output_path: Path) -> bool:
    """Download video from URL"""
    try:
        print(f"  Downloading from {url}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  ✓ Downloaded: {output_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download: {e}")
        return False


def upload_broll_to_api(video_files: list) -> dict:
    """Upload B-Roll videos to the API"""
    print(f"\n📤 Uploading {len(video_files)} B-Roll videos to API...")
    
    files = []
    for video_path in video_files:
        if not video_path.exists():
            print(f"  ✗ File not found: {video_path}")
            continue
        
        files.append(("files", (video_path.name, open(video_path, "rb"), "video/mp4")))
    
    if not files:
        print("  ✗ No valid video files to upload")
        return None
    
    try:
        response = requests.post(f"{API_BASE}/upload/broll", files=files)
        response.raise_for_status()
        result = response.json()
        
        print(f"  ✓ Indexed {result.get('clips_indexed', 0)} clips")
        print(f"  ⏱ Processing time: {result.get('processing_time', 0):.1f}s")
        
        return result
    except Exception as e:
        print(f"  ✗ Upload failed: {e}")
        return None
    finally:
        # Close file handles
        for _, (_, file_obj, _) in files:
            file_obj.close()


def process_aroll(aroll_path: Path) -> str:
    """Process A-Roll video and return job_id"""
    print(f"\n🎬 Processing A-Roll: {aroll_path.name}...")
    
    if not aroll_path.exists():
        print(f"  ✗ File not found: {aroll_path}")
        return None
    
    try:
        with open(aroll_path, "rb") as f:
            files = {"aroll": (aroll_path.name, f, "video/mp4")}
            response = requests.post(f"{API_BASE}/process", files=files)
            response.raise_for_status()
            
        result = response.json()
        job_id = result.get("job_id")
        print(f"  ✓ Job started: {job_id}")
        return job_id
        
    except Exception as e:
        print(f"  ✗ Processing failed: {e}")
        return None


def monitor_job(job_id: str, max_wait: int = 600):
    """Monitor processing job until completion"""
    print(f"\n⏳ Monitoring job: {job_id}")
    
    start_time = time.time()
    last_progress = -1
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{API_BASE}/status/{job_id}")
            response.raise_for_status()
            status = response.json()
            
            progress = status.get("progress", 0)
            message = status.get("message", "")
            job_status = status.get("status", "unknown")
            
            # Print progress update if changed
            if progress != last_progress:
                print(f"  [{progress}%] {message}")
                last_progress = progress
            
            if job_status == "complete":
                print(f"\n✅ Processing complete!")
                
                # Show stats
                stats = status.get("stats", {})
                if stats:
                    print(f"\n📊 Matching Statistics:")
                    print(f"  Total segments: {stats.get('total_segments', 0)}")
                    print(f"  Matched segments: {stats.get('matched_segments', 0)}")
                    print(f"  Match rate: {stats.get('match_rate', 0)*100:.1f}%")
                    print(f"  Avg similarity: {stats.get('avg_similarity', 0):.3f}")
                
                return True
                
            elif job_status == "error":
                error = status.get("error", "Unknown error")
                print(f"\n❌ Processing failed: {error}")
                return False
            
            time.sleep(2)  # Poll every 2 seconds
            
        except Exception as e:
            print(f"  ✗ Status check failed: {e}")
            time.sleep(2)
    
    print(f"\n⏱ Timeout reached ({max_wait}s)")
    return False


def download_result(job_id: str, output_dir: Path):
    """Download the processed video"""
    print(f"\n💾 Downloading result...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"result_{job_id}.mp4"
    
    try:
        response = requests.get(f"{API_BASE}/download/{job_id}", stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"  ✓ Saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return None


def main():
    """Main test workflow"""
    print("=" * 80)
    print("🧪 Semantic A-Roll/B-Roll Engine - JSON Test")
    print("=" * 80)
    
    # Check API health
    try:
        response = requests.get(f"{API_BASE.replace('/api', '')}/", timeout=5)
        if not response.ok:
            print("❌ API is not responding. Please start the backend server:")
            print("   uv run uvicorn backend.main:app --reload")
            return
        print("✅ API is online\n")
    except:
        print("❌ Cannot connect to API. Please start the backend server:")
        print("   uv run uvicorn backend.main:app --reload")
        return
    
    # Load JSON
    json_path = Path(JSON_FILE)
    if not json_path.exists():
        print(f"❌ JSON file not found: {json_path}")
        print("Please update the JSON_FILE path in the script")
        return
    
    print(f"📄 Loading video data from: {json_path}")
    video_data = load_video_json(json_path)
    
    # Parse JSON structure - user's format: {"a_roll": {...}, "b_rolls": [...]}
    broll_videos = []
    aroll_video = None
    
    if "a_roll" in video_data:
        aroll_video = video_data["a_roll"]
        print(f"  ✓ Found A-Roll: {aroll_video.get('metadata', 'No description')[:80]}...")
    
    if "b_rolls" in video_data:
        broll_videos = video_data["b_rolls"]
        print(f"  ✓ Found {len(broll_videos)} B-Roll videos")
        for i, broll in enumerate(broll_videos[:3], 1):  # Show first 3
            print(f"     {i}. {broll.get('id', f'broll_{i}')}: {broll.get('metadata', '')[:60]}...")
        if len(broll_videos) > 3:
            print(f"     ... and {len(broll_videos) - 3} more")
    
    # Create temp directory for downloads
    temp_dir = Path("./test_videos")
    temp_dir.mkdir(exist_ok=True)
    
    # Download B-Roll videos (if URLs provided)
    broll_files = []
    for i, video in enumerate(broll_videos):
        if isinstance(video, str):
            # Assume it's a file path
            video_path = Path(video)
            if video_path.exists():
                broll_files.append(video_path)
        elif isinstance(video, dict):
            if "path" in video and Path(video["path"]).exists():
                broll_files.append(Path(video["path"]))
            elif "url" in video:
                output_path = temp_dir / f"broll_{i}.mp4"
                if download_video(video["url"], output_path):
                    broll_files.append(output_path)
    
    if not broll_files:
        print("❌ No valid B-Roll videos found")
        return
    
    # Upload B-Roll
    result = upload_broll_to_api(broll_files)
    if not result:
        print("❌ B-Roll upload failed")
        return
    
    # Process A-Roll (if provided)
    if aroll_video:
        aroll_path = None
        
        if isinstance(aroll_video, str):
            aroll_path = Path(aroll_video)
        elif isinstance(aroll_video, dict):
            if "path" in aroll_video:
                aroll_path = Path(aroll_video["path"])
            elif "url" in aroll_video:
                aroll_path = temp_dir / "aroll.mp4"
                if not download_video(aroll_video["url"], aroll_path):
                    print("❌ A-Roll download failed")
                    return
        
        if aroll_path and aroll_path.exists():
            job_id = process_aroll(aroll_path)
            if job_id:
                success = monitor_job(job_id)
                if success:
                    output_dir = Path("./test_outputs")
                    download_result(job_id, output_dir)
    else:
        print("\n✅ B-Roll indexing complete!")
        print("   You can now process A-Roll videos through the UI:")
        print("   http://localhost:8501")
    
    print("\n" + "=" * 80)
    print("✅ Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
