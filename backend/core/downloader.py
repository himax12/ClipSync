"""Helper module for downloading videos from URLs"""

import requests
from pathlib import Path
from typing import Optional, List, Tuple
import asyncio
import aiohttp


def download_video(url: str, output_path: Path, timeout: int = 300) -> bool:
    """
    Download video from URL (synchronous)
    
    Args:
        url: Video URL
        output_path: Where to save the video
        timeout: Request timeout in seconds
    
    Returns:
        True if successful, False otherwise
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
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download from {url}: {e}")
        return False


async def download_video_async(url: str, output_path: Path, timeout: int = 300) -> bool:
    """
    Download video from URL (async for parallelism)
    
    First Principles:
    - I/O bound operation: Waiting for network, not CPU
    - Async allows multiple downloads simultaneously
    - N videos in parallel: ~same time as 1 video
    
    Args:
        url: Video URL
        output_path: Where to save the video
        timeout: Request timeout in seconds
    
    Returns:
        True if successful, False otherwise
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
