from dataclasses import dataclass
from typing import Callable
import numpy as np
import mujoco

@dataclass(frozen=True)
class DynamicsParams:
    dyn: str
    f: Callable
    f_np: Callable
    nx: int
    nu: int
    U_lim: np.ndarray
    
@dataclass(frozen=True)
class MuJoCoDynamicsParams(DynamicsParams):
    mj_model: mujoco.MjModel
    mj_data: mujoco.MjData