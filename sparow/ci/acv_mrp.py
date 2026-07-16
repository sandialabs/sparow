import numpy as np
from scipy import stats

# from .standard_mrp import StandardMRP
from .scenario_sampler import ScenarioSampler

class ACVMRP:
    """
    Approximate Control Variate Multiple Replications Procedure (ACV-MRP).
    For estimating the upper bound on the optimality gap of a candidate first-stage solution xhat.
    Uses a low-fidelity model as a control variate to reduce variance in the optimality gap estimator,
    """

    def __init__(self, problem_adapter, scenarios, options):
        self.problem_adapter = problem_adapter
        self.scenarios = scenarios
        self.options = options

        # Validate whether the problem adapter supports ACV-MRP algorithm
        self._validate_acv_support()

        # Validate the full historical / finite scenario population once
        self.problem_adapter.validate_scenario_population(self.scenarios)

        self.sampler = ScenarioSampler(
            scenarios=scenarios,
            seed=options.seed,
            with_replacement=options.with_replacement,
        )

    def _validate_acv_support(self):
        if not self.problem_adapter.supports_acv():
            raise ValueError("Problem adapter does not support ACV-MRP")
        
    def run(self, xhat):

        if self.options.m < 2:
            raise ValueError("ACV-MRP requires m >= 2 to estimate sample variance/covariance.")
          
        if self.options.verbose:
            print(f"Running ACV-MRP with m={self.options.m}, M={self.options.M}, n={self.options.n}")

        # Step 1: Paired replications (k= 1...m)
        paired_replications = self._run_paired_replications(xhat)

        # Step 2: Additional LF-only replications (k = m+1 ... m+M)
        lf_only_replications = self._run_lf_only_replications(xhat)

        # Step 3: Compute ACV statistics
        results = self._compute_acv_statistics(paired_replications, lf_only_replications)

        # reset program state at the end of run for safety
        self.problem_adapter.set_active_fidelity("high")

        return results

    def _run_paired_replications(self, xhat):
        opts = self.options

        paired = []
        
        for rep_id in range(opts.m):
            if opts.verbose:
                print(f"Running paired ACV-MRP replication {rep_id + 1}/{opts.m}")

            # STEP 1 - draw a batch of n iid scenarios
            sampled_scenarios = self.sampler.draw_scenarios(
                    n=opts.n,
                    replication_id=rep_id,
                )
            
            # Validate the format of the sampled scenarios
            self.problem_adapter.validate_scenario_population(sampled_scenarios)

            model_data_k = self.problem_adapter.build_model_data(sampled_scenarios)

            # Evaluate both HF and LF models on same batch of scenarios
            hf_result = self._evaluate_fidelity(xhat, model_data_k, "high")
            lf_result = self._evaluate_fidelity(xhat, model_data_k, "low")

            if opts.verbose:
                print(f"Gap estimate for high-fidelity paired replication {rep_id + 1}: F_nk = {hf_result['gap_estimate']}")
                print(f"Gap estimate for low-fidelity paired replication {rep_id + 1} : G_nk = {lf_result['gap_estimate']}")

            paired.append({
                    "F_nk": hf_result["gap_estimate"],
                    "G_nk": lf_result["gap_estimate"],
                    "sampled_indices": [s["Population_Index"] for s in sampled_scenarios],
                    "scenarios": sampled_scenarios
                })
        return paired
    
    def _run_lf_only_replications(self, xhat):
        """Run M additional LF-only replications."""
        if self.options.M == 0:
            return []
        
        opts = self.options

        lf_only = []

        for rep_id in range(opts.m, opts.m + opts.M):
            if opts.verbose:
                print(f"Running LF-only replication {rep_id + 1 - opts.m}/{opts.M}")

            # STEP 1 - draw a batch of n iid scenarios
            sampled_scenarios = self.sampler.draw_scenarios(
                n=opts.n,
                replication_id=rep_id
            )

            # Validate the format of the sampled scenarios
            self.problem_adapter.validate_scenario_population(sampled_scenarios)

            model_data_k = self.problem_adapter.build_model_data(sampled_scenarios)

            # Evaluate only the LF model on this batch of scenarios
            lf_result = self._evaluate_fidelity(xhat, model_data_k, "low")

            print(f"Gap estimate for low-fidelity additional replication {rep_id + 1} : G_nk = {lf_result['gap_estimate']}")

            lf_only.append({
                "G_nk": lf_result["gap_estimate"],
                "sampled_indices": [s["Population_Index"] for s in sampled_scenarios],
                "scenarios": sampled_scenarios
            })

        return lf_only
    
    def _evaluate_fidelity(self, xhat, model_data, fidelity):
        """Wrapper that handles both standard and ACV adapters."""

        # Tell the adapter which fidelity to use
        self.problem_adapter.set_active_fidelity(fidelity)

        # STEP 1 - draw a batch of n iid scenarios
        # This is done in _run_paired_replications and was used to create model_data argument

        # STEP 2 - solve SAA problem on this replication sample
        # This calls build_stochastic_program internally with the correct model fidelity
        solved_saa = self.problem_adapter.solve_extensive_form(
            model_data=model_data,
            solver_name=self.options.solver_name,
            solver_options=self.options.solver_options,
        )

        saa_optimal_value = self.problem_adapter.get_objective_value(solved_saa)

        # STEP 3 - evaluate fixed candidate xhat on same set of scenarios
        # This calls build_stochastic_program internally with the correct model fidelity
        xhat_value = self.problem_adapter.evaluate_first_stage_solution(
            xhat=xhat,
            model_data=model_data,
            solver_name=self.options.solver_name,
            solver_options=self.options.solver_options,
        )

        gap_raw = xhat_value - saa_optimal_value

        # Allow small absolute or relative numerical error based on the scale of 
        # the objective values
        tol = max(1e-10, 1e-12 * max(1.0, abs(xhat_value), abs(saa_optimal_value)))

        if gap_raw < -tol:
            raise RuntimeError(
                f"Gap estimate is significantly negative: {gap_raw}. "
                f"xhat_value={xhat_value}, saa_optimal_value={saa_optimal_value}"
            )
        
        gap_estimate = max(0.0, gap_raw)

        return {
            "gap_estimate": gap_estimate,
            "xhat_value": xhat_value,
            "saa_optimal_value": saa_optimal_value,
            "fidelity": fidelity
        }
    
    def _compute_acv_statistics(self, paired_reps, lf_only_reps):
        """Compute ACV estimator and confidence interval from replication results."""
        opts = self.options

        # Extract optimality gap estimates from paired and LF-only replications
        F_values = np.array([rep["F_nk"] for rep in paired_reps], dtype=float)
        G_paired_values = np.array([rep["G_nk"] for rep in paired_reps], dtype=float)
        G_all_values = np.concatenate([
            G_paired_values,
            np.array([rep["G_nk"] for rep in lf_only_reps], dtype=float)
        ]) if lf_only_reps else G_paired_values

        # Compute sample means
        F_bar = float(np.mean(F_values))
        G_bar_paired = float(np.mean(G_paired_values))
        G_bar_all = float(np.mean(G_all_values))

        # Compute the paired sample variance and covariance estimates
        s_F_sq = float(np.var(F_values, ddof=1))
        s_G_sq_paired = float(np.var(G_paired_values, ddof=1))
        s_FG = float(np.cov(F_values, G_paired_values, ddof=1)[0, 1]) # [0,1] because np.cov returns a 2x2 matrix

        # Estimate the sample correlation (rho) and control variate coefficient (alpha)
        rho_hat = s_FG / (np.sqrt(s_F_sq) * np.sqrt(s_G_sq_paired)) if s_G_sq_paired > 0 else 0.0
        alpha_hat = s_FG / s_G_sq_paired if s_G_sq_paired > 0 else 0.0

        # Form ACV point estimator: 
        F_acv = F_bar + alpha_hat * (G_bar_all - G_bar_paired)

        # Compute the plug-in variance estimate for ACV estimator
        constant = opts.M / (opts.m * (opts.m + opts.M)) # M / m(m+M)
        expr1 = s_F_sq / opts.m
        expr2 = (alpha_hat ** 2) * s_G_sq_paired * constant
        expr3 = -2.0 * alpha_hat * s_FG * constant

        var_acv = expr1 + expr2 + expr3
        var_acv = max(float(var_acv), 0.0)

        standard_error_acv = float(np.sqrt(var_acv))
        z_statistic = float(stats.norm.ppf(1.0 - opts.alpha))
        half_width = z_statistic * standard_error_acv

        ci_lower = 0.0 # we know the optimality gap is non-negative
        ci_upper = max(0.0, F_acv + half_width)

        # Compute variance reduction factor for comparison
        variance_reduction = s_F_sq / var_acv if var_acv > 0 else float('inf')

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

