import numpy as np
from scipy import stats

# TODO: This is specific to minimization problems... Need to adapt to model sense

class StandardMRP:
    """
    Standard Multiple Replications Procedure for estimating the upper bound on
    the optimality gap of a candidate first-stage solution xhat.

    For each replication k = 1,...,m:
        1. draw a batch of n iid scenarios,
        2. solve the Sample Average Approximation problem to get x_n^{k*},
        3. evaluate xhat on the same set of scenarios,
        4. compute estimate of upper bound on optimality gap F_{n,k}(xhat).
    """

    def __init__(self, model, options):
        self.model = model
        self.options = options

        # Validate the full finite scenario population once at construction.
        self.model.scenario_population().validate()

    def run(self, xhat):
        opts = self.options

        if opts.m < 2:
            raise ValueError("MRP requires m >= 2 to estimate sample variance.")

        replication_values = []
        sampled_population_indices_by_replication = []

        for rep_id in range(opts.m):
            if opts.verbose:
                print(f"Running MRP replication {rep_id + 1}/{opts.m}")

            # STEP 1 - draw a batch of n iid scenarios
            # We also have the option to draw a precomputed superset of scenarios
            # for each replication, and then use the first n scenarios from that superset
            # when doing nested-sample experiments.
            # This is useful for comparing results across different n values.
            sampled_scenarios = self.model.draw_batch_of_scenarios(
                n=opts.n,
                replication_id=rep_id,
                nested_sampling=opts.nested_sampling,
                precomputed_supersets=opts.precomputed_supersets,
            )

            # Keep track of which population scenarios were sampled.
            # We only use the population index to keep track of
            # which scenarios were sampled in each replication.... this is NOT
            # part of the sampling mechanism
            sampled_population_indices_by_replication.append(
                [s["Population_Index"] for s in sampled_scenarios]
            )

            
            # STEP 2 - solve SAA problem on this replication sample
            # STEP 3 - evaluate fixed candidate xhat on same set of scenarios
            # So this computes the replication-level gap estimate:
            #   candidate value minus SAA optimal value
            rep_result = self.model.replication_gap(
                xhat=xhat,
                sampled_scenarios=sampled_scenarios,
                solver_name=opts.solver_name,
                solver_options=opts.solver_options,
            )

            gap_raw = rep_result["gap_estimate"]
            xhat_value = rep_result["xhat_value"]
            saa_optimal_value = rep_result["saa_optimal_value"]

            # Allow small absolute or relative numerical error based on the scale of
            # the objective values
            tol = max(1e-10, 1e-12 * max(1.0, abs(xhat_value), abs(saa_optimal_value)))

            if gap_raw < -tol:
                raise RuntimeError(
                    f"Gap estimate is significantly negative: {gap_raw}. "
                    f"xhat_value={xhat_value}, saa_optimal_value={saa_optimal_value}"
                )

            F_nk = max(0.0, gap_raw)
            replication_values.append(F_nk)

            if opts.verbose:
                print(f"Gap estimate for replication {rep_id + 1}: F_nk = {F_nk}")

        F = np.array(replication_values, dtype=float)

        point_estimate = float(np.mean(F))
        sample_variance = float(np.var(F, ddof=1))
        sample_std = float(np.std(F, ddof=1))

        t_statistic = float(stats.t.ppf(1.0 - opts.alpha, opts.m - 1))
        standard_error = float(sample_std / np.sqrt(opts.m))
        half_width = float(t_statistic * standard_error)

        ci_lower = 0.0  # we know the optimality gap is non-negative
        ci_upper = point_estimate + half_width

        return {
            "point_estimate": point_estimate,
            "sample_variance": sample_variance,
            "sample_std": sample_std,
            "t_statistic": t_statistic,
            "half_width": half_width,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "replication_values": F,
            "sampled_indices_by_replication": sampled_population_indices_by_replication,
            "n": opts.n,
            "m": opts.m,
            "alpha": opts.alpha,
            "with_replacement": opts.with_replacement,
            "seed": opts.seed,
        }
