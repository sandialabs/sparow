import numpy as np
from scipy import stats

from sparow.ci.scenario_sampler import ScenarioSampler

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

    def __init__(self, problem_adapter, scenarios, options):
        self.problem_adapter = problem_adapter
        self.scenarios = scenarios
        self.options = options

        # Validate the full historical / finite scenario population once
        self.problem_adapter.validate_scenario_population(self.scenarios)

        self.sampler = ScenarioSampler(
            scenarios=scenarios,
            seed=options.seed,
            with_replacement=options.with_replacement,
        )

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
            sampled_scenarios = self.sampler.draw_scenarios(
                n=opts.n,
                replication_id=rep_id,
            )

            # Validate the format of the sampled scenarios
            self.problem_adapter.validate_scenario_population(sampled_scenarios)

            # We only use the population index to keep track of 
            # which scenarios were sampled in each replication.... this is NOT
            # part of the sampling mechanism itssample_std / np.sqrt(opts.m)elf!
            sampled_population_indices_by_replication.append([s["Population_Index"] for s in sampled_scenarios])

            model_data_k = self.problem_adapter.build_model_data(sampled_scenarios)

            # STEP 2 - solve SAA problem on this replication sample
            solved_saa = self.problem_adapter.solve_extensive_form(
                model_data=model_data_k,
                solver_name=opts.solver_name,
                solver_options=opts.solver_options,
            )

            saa_optimal_value = self.problem_adapter.get_objective_value(solved_saa)

            # STEP 3 - evaluate fixed candidate xhat on same set of scenarios
            xhat_value = self.problem_adapter.evaluate_first_stage_solution(
                xhat=xhat,
                model_data=model_data_k,
                solver_name=opts.solver_name,
                solver_options=opts.solver_options,
            )

            # STEP 4
            # This is the replication's estimate of the optimality gap upper bound
            # For minimization:
            # F_{n,k}(xhat) = f_n(xhat) - f_n(x_n^{k*})
            F_nk = xhat_value - saa_optimal_value
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

        ci_lower = 0.0 # we know the optimality gap is non-negative
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