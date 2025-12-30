from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class DMPCBase:
    N: int          # horizon length
    Q: np.ndarray   # state-deviation cost
    R: np.ndarray   # control-effort cost
    H: np.ndarray   # terminal state-deviation cost
    term: bool      # terminal constraint
    
@dataclass(frozen=True)
class DecentralizedParams(DMPCBase):
    mode: str       # decentralized mode (Gauss-Seidel vs Jacobi)
    
@dataclass(frozen=True)
class DistributedParams(DMPCBase):
    pass