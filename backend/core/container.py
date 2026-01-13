"""
ClipSync Service Container - Dependency Injection

First Principles:
- Loose coupling: Components don't know concrete implementations
- Easy testing: Swap real services for mocks
- Configuration: All settings in one place
- Factory pattern: Create services with consistent configuration
"""

import os
from typing import Optional, Protocol, runtime_checkable
from pathlib import Path


# =============================================================================
# ABSTRACT INTERFACES (Protocols)
# =============================================================================

@runtime_checkable
class EditorProtocol(Protocol):
    """Interface for video editors (VLM-based or otherwise)"""
    def create_timeline(self, transcript: str, word_timestamps: list, available_clips: list) -> list:
        ...


@runtime_checkable  
class AudioSensorProtocol(Protocol):
    """Interface for audio transcription services"""
    def extract_audio(self, video_path: Path, output_path: Path) -> None:
        ...
    
    def transcribe_and_align(self, audio_path: Path) -> dict:
        ...


@runtime_checkable
class VisionSensorProtocol(Protocol):
    """Interface for vision/embedding services"""
    def extract_keyframes(self, video_path: Path, output_dir: Path, fps: float) -> list:
        ...
    
    def embed_text(self, text: str):
        ...


# =============================================================================
# SERVICE CONTAINER
# =============================================================================

class ServiceContainer:
    """
    Centralized service container for dependency injection.
    
    Usage:
        container = ServiceContainer()
        editor = container.get_editor()
        audio = container.get_audio_sensor()
    
    For testing:
        container = ServiceContainer()
        container.register_editor(MockEditor())
    """
    
    def __init__(self):
        self._editor: Optional[EditorProtocol] = None
        self._audio_sensor: Optional[AudioSensorProtocol] = None
        self._vision_sensor: Optional[VisionSensorProtocol] = None
        
        # Configuration
        self.config = {
            "project_id": os.getenv("GCP_PROJECT_ID", "firstproject-c5ac2"),
            "location": os.getenv("GCP_LOCATION", "us-central1"),
            "gemini_api_key": os.getenv("GOOGLE_API_KEY"),
            "upload_dir": Path(os.getenv("UPLOAD_DIR", "./uploads")),
            "output_dir": Path(os.getenv("OUTPUT_DIR", "./outputs")),
            "data_dir": Path(os.getenv("DATA_DIR", "./data")),
        }
    
    # =========================================================================
    # REGISTRATION (for testing/swapping implementations)
    # =========================================================================
    
    def register_editor(self, editor: EditorProtocol) -> None:
        """Register a custom editor implementation"""
        self._editor = editor
    
    def register_audio_sensor(self, sensor: AudioSensorProtocol) -> None:
        """Register a custom audio sensor implementation"""
        self._audio_sensor = sensor
    
    def register_vision_sensor(self, sensor: VisionSensorProtocol) -> None:
        """Register a custom vision sensor implementation"""
        self._vision_sensor = sensor
    
    # =========================================================================
    # FACTORY METHODS (lazy initialization)
    # =========================================================================
    
    def get_editor(self) -> EditorProtocol:
        """Get or create the video editor"""
        if self._editor is None:
            from backend.core.autonomous_editor import AutonomousEditor
            self._editor = AutonomousEditor(api_key=self.config["gemini_api_key"])
        return self._editor
    
    def get_audio_sensor(self) -> AudioSensorProtocol:
        """Get or create the audio sensor"""
        if self._audio_sensor is None:
            from backend.core.audio_sensor_vertex import AudioSensor
            self._audio_sensor = AudioSensor(
                project_id=self.config["project_id"],
                location=self.config["location"]
            )
        return self._audio_sensor
    
    def get_vision_sensor(self) -> VisionSensorProtocol:
        """Get or create the vision sensor"""
        if self._vision_sensor is None:
            from backend.core import VisionSensor
            self._vision_sensor = VisionSensor(
                project_id=self.config["project_id"],
                location=self.config["location"]
            )
        return self._vision_sensor


# =============================================================================
# GLOBAL CONTAINER INSTANCE
# =============================================================================

# Singleton container for the application
_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """Get the global service container (creates if needed)"""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def reset_container() -> None:
    """Reset the container (useful for testing)"""
    global _container
    _container = None
