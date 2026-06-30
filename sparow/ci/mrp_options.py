from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class MRPOptions:
    """
    Options for the standard Multiple Replications Procedure algorithm.
    """
    n: int                              # sample size per replication
    m: int                              # number of replications
    alpha: float = 0.05                 # confidence level is 1 - alpha
    seed: int = 12345                   # base random seed for entire MRP algorithm run
    with_replacement: bool = True       # bootstrap sampling from finite set of population scenarios
    solver_name: str = "highs"
    solver_options: Optional[Dict[str, Any]] = None
    verbose: bool = True