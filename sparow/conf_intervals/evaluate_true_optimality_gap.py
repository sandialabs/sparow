# TODO: This is specific to minimization problems. Need to adapt to model sense? Or we
# can just tell users to convert their maximization problems into minimization problems by
# multiplying objective by -1.


class TrueOptimalityGapEvaluator:
    """
    Solves the full historical-scenario EF and evaluates a candidate xhat
    on that same full scenario set.

    This is useful when the user wants an exact finite-population benchmark.
    """

    def __init__(self, model, solver_name, solver_options=None):
        """
        Parameters
        ----------
        model : SPModelWrapperforUQ
            One single-fidelity stochastic-program wrapper.
        solver_name : str
            Solver used for the extensive-form solve and fixed-solution evaluation.
        solver_options : dict, optional
            Optional solver settings.
        """
        self.model = model
        self.solver_name = solver_name
        self.solver_options = solver_options

        # Validate the full scenario population once up front.
        self.model.scenario_population().validate()

    def full_scenarios(self):
        """
        Return the full finite scenario population.
        """
        return self.model.scenario_population().scenarios()

    def compute_true_optimal_value(self):
        """
        Solve the full extensive form over the complete finite scenario population.

        Returns
        -------
        float
            The exact finite-population optimal value.
        """
        scenarios = self.full_scenarios()
        solved_full = self.model.solve_saa(
            sampled_scenarios=scenarios,
            solver_name=self.solver_name,
            solver_options=self.solver_options,
        )
        return self.model.get_objective_value(solved_full)

    def evaluate_xhat(self, xhat):
        """
        Evaluate a fixed candidate solution on the complete finite scenario population.

        Parameters
        ----------
        xhat : dict
            Candidate first-stage solution.

        Returns
        -------
        float
            Objective value of xhat on the full finite scenario population.
        """
        scenarios = self.full_scenarios()
        return self.model.evaluate_xhat(
            xhat=xhat,
            sampled_scenarios=scenarios,
            solver_name=self.solver_name,
            solver_options=self.solver_options,
        )

    def compute_true_gap(self, xhat):
        """
        Compute the exact finite-population optimality gap.

        Parameters
        ----------
        xhat : dict
            Candidate first-stage solution.

        Returns
        -------
        dict
            Keys:
              - true_optimal_value
              - xhat_true_value
              - true_gap
        """
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
