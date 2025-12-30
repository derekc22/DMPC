from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class XMLParams:
    name: str
    agent_xml: str
    actuator_xml: str
    gravity: float              # gravity
    dt: float                   # timestep
    M: int                      # number of agents
    nq: int
    q0_val: np.ndarray          # MuJoCo initial states
    obs: np.ndarray             # obstacle positions and radii