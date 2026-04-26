from dataclasses import dataclass
from typing import Optional, Any
import mujoco

@dataclass
class VisualizationParams:
    presets: dict
    track: bool
    show_world_csys: bool
    show_body_csys: bool
    vid_width: int
    vid_height: int
    vid_fps: float
    enable_viewer: bool

    viewer: Any = None
    video_cam: mujoco.MjvCamera = None
    video_renderer: mujoco.Renderer = None
    frames: list = None
    next_frame_time: float = 0.0

