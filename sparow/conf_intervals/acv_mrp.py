import numpy as np
from scipy import stats

from .scenario_sampler import ScenarioSampler


class ACVMRP:
    """
    Approximate Control Variate Multiple Replications Procedure (ACV-MRP).

    This version works directly with two stochastic-program model
    wrappers:
      - one high-fidelity model,
      - one low-fidelity model.

    The first m replications are paired: both models are evaluated on the
    same sampled scenario batch. The next M replications evaluate only the
    low-fidelity model on additional independent batches.
    """

    def __init__(self, hf_model, lf_model, options):
        self.hf_model = hf_model
        self.lf_model = lf_model
        self.options = options

        # Validate the scenario populations once at construction.
        self.hf_model.scenario_population().validate()
        self.lf_model.scenario_population().validate()


    def run(self, xhat):

        if self.options.m < 2:
            raise ValueError("ACV-MRP requires m >= 2 to estimate sample variance/covariance.")

        if self.options.verbose:
            print(f"Running ACV-MRP with m={self.options.m}, M={self.options.M}, n={self.options.n}")
            print(f"Using precomputed superset of scenarios for nested sampling scheme: {self.options.nested_sampling}")

        # Step 1: Paired replications (k= 1...m)
        paired_replications = self._run_paired_replications(xhat)

        # Step 2: Additional LF-only replications (k = m+1 ... m+M)
        lf_only_replications = self._run_lf_only_replications(xhat)

        # Step 3: Compute ACV statistics
        results = self._compute_acv_statistics(
            paired_replications, lf_only_replications
        )

        return results

    def _draw_batch(self, replication_id):
        """
        Draw one shared scenario batch.

        The same batch is used for both HF and LF in paired replications so that
        the induced correlation can be exploited by the control variate.
        """
        opts = self.options

        return self.hf_model.draw_batch_of_scenarios(
            n=opts.n,
            replication_id=replication_id,
            nested_sampling=opts.nested_sampling,
            precomputed_supersets=opts.precomputed_supersets,
        )

    def _run_paired_replications(self, xhat):
        """
        Run the first m paired replications.

        Each paired replication evaluates both HF and LF on the same scenario batch.
        """

        opts = self.options
        paired = []

        for rep_id in range(opts.m):

            if opts.verbose:
                print(f"Running paired ACV-MRP replication {rep_id + 1}/{opts.m}")

            sampled_scenarios = self._draw_batch(rep_id)

            hf_result = self.hf_model.replication_gap(
                xhat=xhat,
                sampled_scenarios=sampled_scenarios,
                solver_name=opts.solver_name,
                solver_options=opts.solver_options,
            )

            lf_result = self.lf_model.replication_gap(
                xhat=xhat,
                sampled_scenarios=sampled_scenarios,
                solver_name=opts.solver_name,
                solver_options=opts.solver_options,
            )

            if opts.verbose:
                print(
                    f"Gap estimate for high-fidelity paired replication {rep_id + 1}: F_nk = {hf_result['gap_estimate']}"
                )
                print(
                    f"Gap estimate for low-fidelity paired replication {rep_id + 1} : G_nk = {lf_result['gap_estimate']}"
                )

            paired.append(
                {
                    "F_nk": hf_result["gap_estimate"],
                    "G_nk": lf_result["gap_estimate"],
                    "sampled_indices": [
                        s["Population_Index"] for s in sampled_scenarios
                    ],
                    "scenarios": sampled_scenarios,
                }
            )
        return paired

    def _run_lf_only_replications(self, xhat):
        """
        Run the additional M LF-only replications.
        """
        if self.options.M == 0:
            return []

        opts = self.options

        lf_only = []

        for rep_id in range(opts.m, opts.m + opts.M):

            if opts.verbose:
                print(f"Running LF-only replication {rep_id + 1 - opts.m}/{opts.M}")

            sampled_scenarios = self._draw_batch(rep_id)

            lf_result = self.lf_model.replication_gap(
                xhat=xhat,
                sampled_scenarios=sampled_scenarios,
                solver_name=opts.solver_name,
                solver_options=opts.solver_options,
            )

            if opts.verbose:
                print(
                    f"Gap estimate for low-fidelity additional replication {rep_id + 1} : G_nk = {lf_result['gap_estimate']}"
                )

            lf_only.append(
                {
                    "G_nk": lf_result["gap_estimate"],
                    "sampled_indices": [
                        s["Population_Index"] for s in sampled_scenarios
                    ],
                    "scenarios": sampled_scenarios,
                }
            )

        return lf_only

    def _compute_acv_statistics(self, paired_reps, lf_only_reps):
        """
        Compute the ACV point estimator and its confidence interval.
        """
        opts = self.options

        # Extract optimality gap estimates from paired and LF-only replications
        F_values = np.array([rep["F_nk"] for rep in paired_reps], dtype=float)
        G_paired_values = np.array([rep["G_nk"] for rep in paired_reps], dtype=float)
        G_all_values = (
            np.concatenate(
                [
                    G_paired_values,
                    np.array([rep["G_nk"] for rep in lf_only_reps], dtype=float),
                ]
            )
            if lf_only_reps
            else G_paired_values
        )

        # Compute sample means
        F_bar = float(np.mean(F_values))
        G_bar_paired = float(np.mean(G_paired_values))
        G_bar_all = float(np.mean(G_all_values))

        # Compute the paired sample variance and covariance estimates
        s_F_sq = float(np.var(F_values, ddof=1))
        s_G_sq_paired = float(np.var(G_paired_values, ddof=1))
        s_FG = float(
            np.cov(F_values, G_paired_values, ddof=1)[0, 1]
        )  # [0,1] because np.cov returns a 2x2 matrix

        # Estimate the sample correlation (rho) and control variate coefficient (alpha)
        rho_hat = (
            s_FG / (np.sqrt(s_F_sq) * np.sqrt(s_G_sq_paired))
            if s_G_sq_paired > 0
            else 0.0
        )
        alpha_hat = s_FG / s_G_sq_paired if s_G_sq_paired > 0 else 0.0

        # Form ACV point estimator:
        F_acv = F_bar + alpha_hat * (G_bar_all - G_bar_paired)

        # Compute the plug-in variance estimate for ACV estimator
        constant = opts.M / (opts.m * (opts.m + opts.M))  # M / m(m+M)
        expr1 = s_F_sq / opts.m
        expr2 = (alpha_hat**2) * s_G_sq_paired * constant
        expr3 = -2.0 * alpha_hat * s_FG * constant

        var_acv = expr1 + expr2 + expr3
        var_acv = max(float(var_acv), 0.0)

        standard_error_acv = float(np.sqrt(var_acv))
        z_statistic = float(stats.norm.ppf(1.0 - opts.alpha))
        half_width = z_statistic * standard_error_acv

        ci_lower = 0.0  # we know the optimality gap is non-negative
        ci_upper = max(0.0, F_acv + half_width)

        # Compute variance reduction factor for comparison
        # This is the benefit provided by the additional M low-fidelity reps
        variance_reduction = s_F_sq / var_acv if var_acv > 0 else float("inf")

        return {
            "point_estimate": F_acv,
            "point_estimate_hf_only": F_bar,  # For comparison
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "half_width": half_width,
            "control_variate_coefficient": alpha_hat,
            "sample_correlation": rho_hat,
            "sample_variance_F": s_F_sq,
            "sample_variance_G_paired": s_G_sq_paired,
            "sample_covariance_FG": s_FG,
            "variance_acv_estimator": float(var_acv),
            "standard_error_acv": standard_error_acv,
            "z_statistic": z_statistic,
            "variance_reduction_factor": variance_reduction,
            # Replication data
            "F_values": F_values,
            "G_paired_values": G_paired_values,
            "G_all_values": G_all_values,
            "paired_replications": paired_reps,
            "lf_only_replications": lf_only_reps,
            # Configuration
            "n": opts.n,
            "m": opts.m,
            "M": opts.M,
            "alpha": opts.alpha,
            "with_replacement": opts.with_replacement,
            "seed": opts.seed,
        }
