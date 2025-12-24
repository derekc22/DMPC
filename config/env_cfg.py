from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class EnvBase:
    T: float                    # total sim time
    dt: float                   # timestep
    M: int                      # number of agents
    d_min: float                # minimum separation distance
    x0_val: np.ndarray          # initial states
    obs: np.ndarray             # obstacle positions and radii
    sigma: float                # disturbance stddev
    
@dataclass(frozen=True)
class EnvParams(EnvBase):
    xf_val: np.ndarray          # final states
    
@dataclass(frozen=True)
class LeaderEnvParams(EnvBase):
    xf_val_leader: np.ndarray   # final leader state
    
@dataclass(frozen=True)
class RendezvousEnvParams(EnvBase):
    pass