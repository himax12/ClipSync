"""
Video Downloader Module with Validation

First Principles:
- HTTP 200 doesn't mean valid video (could be HTML error page)
- FFprobe can detect corrupted/invalid video files
- Validate BEFORE passing to processing pipeline
"""

import subprocess
import requests
from pathlib import Path
from typing import Optional, List, Tuple
import asyncio
import aiohttp


def validate_video(video_path: Path) -> bool:
    """
    Validate that a file is a valid video using ffprobe.
    
    First Principles:
    - HTTP 200 response could still be HTML error page
    - Binary file could be corrupted/truncated
    - Only ffprobe can confirm it's a valid video
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if valid video, False otherwise
    """
    if not video_path.exists():
        return False
    
    # File size check (videos should be > 10KB minimum)
    if video_path.stat().st_size < 10 * 1024:
        print(f"  ✗ {video_path.name}: File too small ({video_path.stat().st_size} bytes)")
        return False
    
    try:
        # Use ffprobe to validate video structure
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",  # Only show errors
                "-select_streams", "v:0",  # Check video stream
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # If ffprobe finds a video stream, it's valid
        if result.returncode == 0 and "video" in result.stdout:
            return True
        else:
            print(f"  ✗ {video_path.name}: Not a valid video file")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ {video_path.name}: FFprobe timeout (possibly corrupted)")
        return False
    except FileNotFoundError:
        print(f"  ⚠️ FFprobe not found - skipping validation")
        return True  # Assume valid if ffprobe not installed
    except Exception as e:
        print(f"  ✗ {video_path.name}: Validation error: {e}")
        return False


def download_video(url: str, output_path: Path, timeout: int = 300) -> bool:
    """
    Download video from URL with validation (synchronous)
    
    Args:
        url: Video URL
        output_path: Where to save the video
        timeout: Request timeout in seconds
    
    Returns:
        True if successful AND valid video, False otherwise
    """
    try:
        print(f"  Downloading from {url}...")
        
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download in chunks
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        print(f"  ✓ Downloaded: {output_path.name} ({downloaded / 1024 / 1024:.1f} MB)")
        
        # VALIDATE: Is it actually a video?
        if not validate_video(output_path):
            print(f"  ✗ {output_path.name}: Downloaded but not a valid video")
            output_path.unlink(missing_ok=True)  # Delete invalid file
            return False
        
        print(f"  ✓ Validated: {output_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download from {url}: {e}")
        return False


async def download_video_async(url: str, output_path: Path, timeout: int = 300) -> bool:
    """
    Download video from URL with validation (async for parallelism)
    
    First Principles:
    - I/O bound operation: Waiting for network, not CPU
    - Async allows multiple downloads simultaneously
    - N videos in parallel: ~same time as 1 video
    - MUST validate after download (HTTP 200 ≠ valid video)
    
    Args:
        url: Video URL
        output_path: Where to save the video
        timeout: Request timeout in seconds
    
    Returns:
        True if successful AND valid video, False otherwise
    """
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(output_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                        downloaded += len(chunk)
        
        print(f"  ✓ {output_path.name}: {downloaded / 1024 / 1024:.1f} MB")
        
        # VALIDATE: Is it actually a video?
        if not validate_video(output_path):
            print(f"  ✗ {output_path.name}: Downloaded but not a valid video")
            output_path.unlink(missing_ok=True)  # Delete invalid file
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ {output_path.name} failed: {e}")
        return False


async def download_all_videos_async(video_urls: List[Tuple[str, Path]]) -> List[bool]:
    """
    Download multiple videos in parallel
    
    First Principles:
    - Each download is I/O bound (waiting for network)
    - CPU mostly idle during downloads
    - Run all downloads concurrently → N× faster!
    
    Args:
        video_urls: List of (url, output_path) tuples
    
    Returns:
        List of success/failure booleans
    """
    print(f"📥 Downloading {len(video_urls)} videos in parallel...")
    
    tasks = [download_video_async(url, path) for url, path in video_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to False
    results = [r if isinstance(r, bool) else False for r in results]
    
    successful = sum(results)
    print(f"✓ Downloaded {successful}/{len(video_urls)} videos successfully")
    
    return results
