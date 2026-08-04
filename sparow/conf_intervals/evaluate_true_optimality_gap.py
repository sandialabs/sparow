# TODO: This is specific to minimization problems. Need to adapt to model sense? Or we
# can just tell users to convert their maximization problems into minimization problems by
# multiplying objective by -1.


class TrueOptimalityGapEvaluator:
    """
    Solves the full historical-scenario EF and evaluates a candidate xhat
    on that same full scenario set.

    This is useful when the user wants an exact finite-population benchmark.
    """

    def __init__(
        self, problem_adapter, scenarios, solver_name="highs", solver_options=None
    ):
        self.problem_adapter = problem_adapter
        self.scenarios = scenarios
        self.solver_name = solver_name
        self.solver_options = solver_options

        # Validate scenarios once
        self.problem_adapter.validate_scenario_population(self.scenarios)

    # STEP 1 - solve the true EF over the full population of historical scenarios
    # to get the true optimal value
    def compute_true_optimal_value(self):
        model_data = self.problem_adapter.build_model_data(self.scenarios)

        solved_full = self.problem_adapter.solve_extensive_form(
            model_data=model_data,
            solver_name=self.solver_name,
            solver_options=self.solver_options,
        )

        return self.problem_adapter.get_objective_value(solved_full)

    # STEP 2 - we are given a candidate solution, xhat... TODO: read in from npy file?
    # STEP 3 - rebuild the full EF over all historical population scenarios and fix xhat
    # STEP 4 - with xhat fixed, solve the recourse EF over model_data, and return the resulting objective value.
    def evaluate_xhat(self, xhat):
        model_data = self.problem_adapter.build_model_data(self.scenarios)

        return self.problem_adapter.evaluate_first_stage_solution(
            xhat=xhat,
            model_data=model_data,
            solver_name=self.solver_name,
            solver_options=self.solver_options,
        )

    # STEP 5 - return true optimality gap
    def compute_true_gap(self, xhat):
        true_optimal_value = self.compute_true_optimal_value()
        xhat_true_value = self.evaluate_xhat(xhat)

        true_gap_raw = xhat_true_value - true_optimal_value

        # Allow small absolute or relative numerical error based on the scale of
        # the objective values
        tol = max(
            1e-10, 1e-12 * max(1.0, abs(xhat_true_value), abs(true_optimal_value))
        )

        if true_gap_raw < -tol:
            raise RuntimeError(
                "Computed true optimality gap is significantly negative, which suggests "
                "more than floating-point error.\n"
                f"xhat_true_value = {xhat_true_value}\n"
                f"true_optimal_value = {true_optimal_value}\n"
                f"true_gap_raw = {true_gap_raw}\n"
                f"tolerance = {tol}"
            )

        true_gap = max(0.0, true_gap_raw)

        return {
            "true_optimal_value": true_optimal_value,
            "xhat_true_value": xhat_true_value,
            "true_gap": true_gap,
        }
