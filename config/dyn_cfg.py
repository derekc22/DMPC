from dataclasses import dataclass
from typing import Callable
import numpy as np
import mujoco

@dataclass(frozen=True)
class DynamicsParams:
    name: str           # dynamics type
    f_plant: Callable   # plant dynamics
    f_true: Callable    # true dynamics
    nx: int             # plant state dimension
    nu: int             # plant input dimension
    u_lim: np.ndarray   # plant input limits
    
@dataclass(frozen=True)
class MuJoCoDynamicsParams(DynamicsParams):
    mj_model: mujoco.MjModel    # MuJoCo model object
    mj_data: mujoco.MjData      # MuJoCo data object