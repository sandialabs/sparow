
class ProgressiveHedgingSolver(object):

    def __init__(self):
        self.rho = 1.5
        self.max_iterations = 100
        # The StochProgram object manages the sub-solver interface.  By default, we assume
        #   the user has initialized the sub-solver within the SP object.
        self.solver_name = None
        self.solver_options = {}
        
    def solve(self, sp, *, rho=None, max_iterations=None, solver=None, solver_options=None):
        #
        # Misc configuration
        #
        if rho:
            self.rho = rho
        if max_iterations:
            self.max_iterations = max_iterations
        if solver:
            self.solver_name = solver
        if solver_options:
            self.solver_options = solver_options
        if self.solver_name:
            sp.set_solver(self.solver_name)

        # Step 2
        results = {}
        M = {}
        for b in sp.bundles:
            M[b] = sp.create_subproblem(b)
            results[b] = sp.solve(M[b], solver_options=self.solver_options)

        #
        # This is a list of shared first-stage variables amongst all bundles.
        # Note: we need to initialize this after we create our initial sub-problems.
        #
        sfs_variables = sp.shared_variables()

        # Step 3
        x_bar = {}
        for x in sfs_variables:
            x_bar[x] = 0.0
            for b in sp.bundles:
                x_bar[x] += sp.bundle_probability[b] * sp.get_variable_value(M[b], x)

        # Step 4
        w = {}
        for b in sp.bundles:
            w[b] = {}
            for x in sfs_variables:
                w[b][x] = self.rho * (sp.get_variable_value(M[b], x) - x_bar[x])

        while k < self.max_iterations:
            # Step 5
            k += 1
            x_bar_prev = x_bar
            w_prev = w

            # Step 6
            results = {}
            M = {}
            for b in sp.bundles:
                M[b] = sp.create_subproblem(b=b, w=w_prev, x_bar=x_bar_prev, rho=self.rho)
                #self.save_results(k, s, results[s], M[s])

            # Step 7
            x_bar = {}
            for x in sfs_variables:
                x_bar[x] = 0.0
                for b in sp.bundles:
                    x_bar[x] += sp.bundle_probability(b) * sp.get_variable_value(x,M[b])

            # Step 8
            w = {}
            for b in sp.bundles:
                w[b] = {}
                for x in sfs_variables:
                    w[b][x] = w_prev[b][x] + self.rho * (sp.get_variable_value(x,M[b]) - x_bar[x])

            # Step 9
            g = 0.0
            for b in sp.bundles:
                g += sp.bundle_probability(b) * self.norm(sp.get_variable_value(x,M[b]) - x_bar[x] for x in sfs_variables)

            # Step 10
            if g < self.convergence_tolerance:
                break

            self.update_rho()

        self.store_results(x_bar=x_bar, w=w, g=g)
            
    def update_rho(self):
        # TODO HERE
        pass

    def store_results(self, *, x_bar, w, g):
        # Abstract
        pass
