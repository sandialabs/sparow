from dataclasses import dataclass
from typing import Optional, Dict, Any

from .mrp_options import MRPOptions


@dataclass
class ACVMRPOptions:
    """
    Options for the Approximate Control Variate Multiple Replications Procedure (ACV-MRP).

    Extends the standard MRP options with additional parameters for:
    - Additional low-fidelity replications (M)
    - PyApprox group ACV integration
    - Control variate specific configuration
    """

    n: int  # Sample size per replication
    m: int  # Number of paired replications (HF+LF)
    M: int  # Additional LF-only replications
    alpha: float = 0.05  # Confidence level is 1 - alpha
    seed: int = 12345  # Base random seed
    with_replacement: bool = (
        True  # Bootstrap sampling from finite set of population scenarios
    )
    solver_name: str = "gurobi_direct"
    solver_options: Optional[Dict[str, Any]] = None
    verbose: bool = False

    # PyApprox integration
    use_pyapprox: bool = False  # Enable PyApprox group ACV
    pyapprox_config: Optional[Dict[str, Any]] = None  # PyApprox configuration

    # ACV-specific parameters
    correlation_estimate: Optional[float] = None  # Initial correlation estimate
    allocation_strategy: str = "optimal"  # PyApprox allocation strategy

    # Optional controls for nested-sample experiments.
    # Default is False (each replication draws its own independent sample of size n)
    nested_sampling: bool = False
    precomputed_supersets: Optional[Dict[int, list]] = (
        None  # key = rep_id, value = list of sampled scenarios of size n_max
    )

    def to_mrp_options(self) -> MRPOptions:
        """
        Convert to standard MRPOptions for compatibility with existing code.

        Returns
        -------
        MRPOptions
            Equivalent standard MRP options (ignoring ACV-specific parameters)
        """

        return MRPOptions(
            n=self.n,
            m=self.m,
            alpha=self.alpha,
            seed=self.seed,
            with_replacement=self.with_replacement,
            solver_name=self.solver_name,
            solver_options=self.solver_options,
            verbose=self.verbose,
            nested_sampling=self.nested_sampling,
            precomputed_supersets=self.precomputed_supersets,
        )
