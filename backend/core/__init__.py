"""Core modules initialization"""

# NOTE: We use audio_sensor_vertex (Vertex AI STT), NOT audio_sensor (WhisperX)
# This avoids the heavy PyTorch/WhisperX dependencies
from .audio_sensor_vertex import AudioSensor
from .vision_sensor import VisionSensor
from .memory_layer import MemoryLayer
from .solver import SemanticSolver
from .actuator import VideoActuator
from .container import ServiceContainer, get_container, reset_container
from .config import ClipSyncConfig, get_config, reset_config

__all__ = [
    "AudioSensor",
    "VisionSensor",
    "MemoryLayer",
    "SemanticSolver",
    "VideoActuator",
    "ServiceContainer",
    "get_container",
    "reset_container",
    "ClipSyncConfig",
    "get_config",
    "reset_config",
]

