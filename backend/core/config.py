"""
ClipSync Configuration Module

First Principles:
- Single source of truth for all settings
- Environment variables override defaults
- Type-safe with validation
- Easy to extend for new settings
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClipSyncConfig:
    """
    Centralized configuration for ClipSync.
    
    All settings are read from environment variables with sensible defaults.
    This eliminates hardcoded values scattered across the codebase.
    
    Usage:
        config = ClipSyncConfig.from_env()
        print(config.gemini_model)  # "gemini-2.0-flash"
    """
    
    # ==========================================================================
    # GCP Settings
    # ==========================================================================
    gcp_project_id: str = "firstproject-c5ac2"
    gcp_location: str = "us-central1"
    
    # ==========================================================================
    # API Keys
    # ==========================================================================
    google_api_key: Optional[str] = None
    
    # ==========================================================================
    # VLM Settings
    # ==========================================================================
    gemini_model: str = "gemini-2.0-flash"
    gemini_endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models"
    vlm_temperature: float = 0.3
    vlm_max_tokens: int = 1000
    vlm_timeout_seconds: int = 90
    vlm_max_retries: int = 3
    vlm_retry_base_delay: float = 1.0
    
    # ==========================================================================
    # Processing Settings
    # ==========================================================================
    keyframe_fps: float = 1.0  # Frames per second for B-Roll extraction
    min_segment_duration: float = 3.0  # Minimum segment length in seconds
    max_segment_duration: float = 10.0  # Maximum segment length in seconds
    min_broll_duration: float = 3.0  # Minimum B-Roll cut duration
    max_broll_duration: float = 8.0  # Maximum B-Roll cut duration
    
    # ==========================================================================
    # Directories
    # ==========================================================================
    upload_dir: Path = field(default_factory=lambda: Path("./uploads"))
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    
    # ==========================================================================
    # Video Settings
    # ==========================================================================
    default_resolution: str = "1920:1080"
    video_codec: str = "libx264"
    video_preset: str = "fast"
    video_crf: int = 23  # Quality (lower = better, 18-28 typical range)
    
    # ==========================================================================
    # Health Check Thresholds
    # ==========================================================================
    min_disk_space_gb: float = 5.0
    min_video_size_kb: int = 10  # Minimum valid video size in KB
    min_image_coverage: float = 0.5  # Warn if <50% of clips have images
    
    @classmethod
    def from_env(cls) -> "ClipSyncConfig":
        """
        Create config from environment variables.
        
        Environment variable names are uppercase versions of field names.
        Example: gemini_model -> GEMINI_MODEL
        """
        return cls(
            # GCP
            gcp_project_id=os.getenv("GCP_PROJECT_ID", cls.gcp_project_id),
            gcp_location=os.getenv("GCP_LOCATION", cls.gcp_location),
            
            # API Keys
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            
            # VLM
            gemini_model=os.getenv("GEMINI_MODEL", cls.gemini_model),
            gemini_endpoint=os.getenv("GEMINI_ENDPOINT", cls.gemini_endpoint),
            vlm_temperature=float(os.getenv("VLM_TEMPERATURE", cls.vlm_temperature)),
            vlm_max_tokens=int(os.getenv("VLM_MAX_TOKENS", cls.vlm_max_tokens)),
            vlm_timeout_seconds=int(os.getenv("VLM_TIMEOUT_SECONDS", cls.vlm_timeout_seconds)),
            vlm_max_retries=int(os.getenv("VLM_MAX_RETRIES", cls.vlm_max_retries)),
            vlm_retry_base_delay=float(os.getenv("VLM_RETRY_BASE_DELAY", cls.vlm_retry_base_delay)),
            
            # Processing
            keyframe_fps=float(os.getenv("KEYFRAME_FPS", cls.keyframe_fps)),
            min_segment_duration=float(os.getenv("MIN_SEGMENT_DURATION", cls.min_segment_duration)),
            max_segment_duration=float(os.getenv("MAX_SEGMENT_DURATION", cls.max_segment_duration)),
            min_broll_duration=float(os.getenv("MIN_BROLL_DURATION", cls.min_broll_duration)),
            max_broll_duration=float(os.getenv("MAX_BROLL_DURATION", cls.max_broll_duration)),
            
            # Directories
            upload_dir=Path(os.getenv("UPLOAD_DIR", "./uploads")),
            output_dir=Path(os.getenv("OUTPUT_DIR", "./outputs")),
            data_dir=Path(os.getenv("DATA_DIR", "./data")),
            
            # Video
            default_resolution=os.getenv("DEFAULT_RESOLUTION", cls.default_resolution),
            video_codec=os.getenv("VIDEO_CODEC", cls.video_codec),
            video_preset=os.getenv("VIDEO_PRESET", cls.video_preset),
            video_crf=int(os.getenv("VIDEO_CRF", cls.video_crf)),
            
            # Health Check
            min_disk_space_gb=float(os.getenv("MIN_DISK_SPACE_GB", cls.min_disk_space_gb)),
            min_video_size_kb=int(os.getenv("MIN_VIDEO_SIZE_KB", cls.min_video_size_kb)),
            min_image_coverage=float(os.getenv("MIN_IMAGE_COVERAGE", cls.min_image_coverage)),
        )
    
    @property
    def gemini_api_url(self) -> str:
        """Full Gemini API URL for the configured model."""
        return f"{self.gemini_endpoint}/{self.gemini_model}:generateContent"
    
    def validate(self) -> list:
        """Validate configuration and return list of issues."""
        issues = []
        
        if not self.google_api_key:
            issues.append("GOOGLE_API_KEY not set")
        
        if self.vlm_temperature < 0 or self.vlm_temperature > 1:
            issues.append(f"VLM_TEMPERATURE must be 0-1, got {self.vlm_temperature}")
        
        if self.video_crf < 0 or self.video_crf > 51:
            issues.append(f"VIDEO_CRF must be 0-51, got {self.video_crf}")
        
        if self.keyframe_fps <= 0:
            issues.append(f"KEYFRAME_FPS must be positive, got {self.keyframe_fps}")
        
        return issues
    
    def print_summary(self):
        """Print configuration summary for debugging."""
        print("\n📋 ClipSync Configuration:")
        print(f"   GCP Project: {self.gcp_project_id}")
        print(f"   VLM Model: {self.gemini_model}")
        print(f"   Keyframe FPS: {self.keyframe_fps}")
        print(f"   Resolution: {self.default_resolution}")
        print(f"   Upload Dir: {self.upload_dir}")


# =============================================================================
# GLOBAL CONFIG INSTANCE
# =============================================================================

_config: Optional[ClipSyncConfig] = None


def get_config() -> ClipSyncConfig:
    """Get the global configuration (creates from env if needed)."""
    global _config
    if _config is None:
        _config = ClipSyncConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset config (useful for testing)."""
    global _config
    _config = None
