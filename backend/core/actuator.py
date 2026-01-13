"""
Video Actuator Module
Frame-perfect video assembly using FFmpeg filter complex
"""

import subprocess
from pathlib import Path
from typing import List, Dict
import json


class VideoActuator:
    @staticmethod
    def assemble_timeline(
        aroll_video: Path,
        timeline_plan: List[Dict],
        output_path: Path,
        resolution: str = "1920:1080"
    ) -> Path:
        """
        Single-pass video rendering using filter_complex
        
        Strategy: Overlay B-Roll clips at precise timestamps using FFmpeg's
        enable filter for frame-perfect timing. Original audio is preserved.
        
        Args:
            aroll_video: Path to A-Roll video (base layer)
            timeline_plan: List of timeline entries from SemanticSolver
            output_path: Output video path
            resolution: Target resolution (width:height)
        
        Returns:
            Path to generated video
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # DEBUG: Show timeline plan
        print(f"\n   📋 Timeline plan received: {len(timeline_plan)} entries")
        for i, entry in enumerate(timeline_plan):
            print(f"      [{i}] {entry.get('aroll_start', 0):.1f}s-{entry.get('aroll_end', 0):.1f}s → {entry.get('broll_name', 'unknown')} (path: {entry.get('broll_clip', 'NONE')[:30] if entry.get('broll_clip') else 'NONE'}...)")
        
        # Build input files list
        inputs = ["-i", str(aroll_video)]
        input_idx = 1
        
        # Map B-Roll files to input indices
        broll_map = {}
        for entry in timeline_plan:
            if entry["broll_clip"] and entry["broll_clip"] not in broll_map:
                inputs.extend(["-i", str(entry["broll_clip"])])
                broll_map[entry["broll_clip"]] = input_idx
                input_idx += 1
        
        print(f"   📋 B-Roll map: {len(broll_map)} unique clips → inputs 1-{input_idx-1}")
        
        # If no B-Roll matches, just copy the A-Roll
        if not broll_map:
            print("No B-Roll matches found. Copying A-Roll as output...")
            subprocess.run([
                "ffmpeg", "-i", str(aroll_video),
                "-c", "copy", "-y", str(output_path)
            ], check=True, capture_output=True)
            return output_path
        
        # Get A-Roll resolution (allow original, but ensure even dimensions for x264)
        aroll_info = VideoActuator.get_video_info(aroll_video)
        raw_width, raw_height = aroll_info["width"], aroll_info["height"]
        
        # Ensure mod-2 resolution (required for yuv420p/libx264)
        width = raw_width if raw_width % 2 == 0 else raw_width - 1
        height = raw_height if raw_height % 2 == 0 else raw_height - 1
        
        if width != raw_width or height != raw_height:
            print(f"⚠️ Adjusted output resolution from {raw_width}x{raw_height} to {width}x{height} (mod-2 requirement)")
        else:
            print(f"   Using A-Roll resolution: {width}x{height}")
        
        # Build filter complex
        filter_parts = []
        
        # 0. Prepare Base: Scale/Crop A-Roll to safe even resolution if needed
        # This ensures the base layer itself is compliant before we overlay anything
        if width != raw_width or height != raw_height:
             base_label = "base_safe"
             filter_parts.append(f"[0:v]crop={width}:{height}:0:0[{base_label}]")
             overlay_chain = base_label
        else:
             overlay_chain = "0:v"

        for i, entry in enumerate(timeline_plan):
            if entry["broll_clip"]:
                broll_idx = broll_map[entry["broll_clip"]]
                start = entry["aroll_start"]
                end = entry["aroll_end"]
                duration = end - start
                
                # ROBUST B-ROLL PROCESSING PIPELINE:
                # 1. loop: Makes input infinite so we never run out of frames (fixes "stagnant")
                # 2. trim: Cuts it to the EXACT duration needed for this slot (fixes "infinite loop" / hang)
                # 3. setpts: Shifts timestamps to match the overlay start time (fixes "not appearing")
                #    using ({start}/TB) to convert seconds to timebase units.
                
                scaled_label = f"b{i}_scaled"
                
                scale_filter = (
                    f"[{broll_idx}:v]"
                    f"loop=loop=-1:size=32767:start=0,"
                    f"trim=duration={duration},"
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                    f"setsar=1,"
                    f"setpts=PTS-STARTPTS+({start}/TB)"
                    f"[{scaled_label}]"
                )
                filter_parts.append(scale_filter)
                
                # Overlay at precise time
                next_label = f"overlay{i}"
                overlay_filter = (
                    f"[{overlay_chain}][{scaled_label}]"
                    f"overlay=enable='between(t,{start},{end})':eof_action=pass"
                    f"[{next_label}]"
                )
                filter_parts.append(overlay_filter)
                overlay_chain = next_label
        
        # Final output format
        filter_parts.append(f"[{overlay_chain}]format=yuv420p[final]")
        filter_complex = ";".join(filter_parts)
        
        print(f"\n🎬 Assembling video with {len(broll_map)} B-Roll clips...")
        print(f"   Filter complexity: {len(filter_parts)} operations")
        print(f"   Timeline entries: {len(timeline_plan)}")
        
        # DEBUG: Print filter_complex to diagnose issues
        print(f"\n   📋 Filter chain:")
        for part in filter_parts:
            print(f"      {part[:80]}...")
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[final]",
            "-map", "0:a",  # Keep original audio from A-Roll
            "-c:v", "libx264",
            "-preset", "fast",  # Balance speed vs compression
            "-crf", "23",  # Quality (lower = better, 23 is default)
            "-c:a", "copy",  # Copy audio without re-encoding
            "-y", str(output_path)
        ]
        
        # Execute FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        print(f"✓ Video assembled: {output_path}")
        return output_path
    
    @staticmethod
    def get_video_info(video_path: Path) -> Dict:
        """Get video metadata using ffprobe"""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Extract key info
        video_stream = next(
            (s for s in data.get("streams", []) if s["codec_type"] == "video"),
            None
        )
        
        if not video_stream:
            raise ValueError("No video stream found")
        
        return {
            "duration": float(data["format"]["duration"]),
            "width": int(video_stream["width"]),
            "height": int(video_stream["height"]),
            "fps": eval(video_stream["r_frame_rate"]),  # "30000/1001" → float
            "codec": video_stream["codec_name"]
        }
