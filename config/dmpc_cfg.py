from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class DMPCBase:
    N: int
    Q: np.ndarray
    R: np.ndarray
    H: np.ndarray
    term: bool
    
@dataclass(frozen=True)
class DecentralizedParams(DMPCBase):
    mode: str
    
@dataclass(frozen=True)
class DistributedParams(DMPCBase):
    pass