
class ProgressiveHedgingSolve(object):

    def create_subproblem(self, args):
        pass

    def solve(self):
        solver = SolverFactory(self.solver_name)

        self.initialize()

        # Step 2
        results = {}
        M = {}
        for b in self.bundles:
            M[b] = self.create_EF(b)
            results[b] = solver.solve(M[b], solver_options=self.solver_options)

        # Step 3
        x_bar = {}
        for x in self.first_stage_variables:
            x_bar[x] = 0.0
            for b in self.bundles:
                x_bar[x] += self.bundle_probability[b] * self.get_variable_value(M[b], x)

        # Step 4
        w = {}
        for b in self.bundles:
            w[b] = {}
            for x in self.first_stage_variables:
                w[b][x] = self.rho * (self.get_variable_value(M[b], x) - x_bar[x])

        while k < max_iterations:
            # Step 5
            k += 1
            x_bar_prev = x_bar
            w_prev = w

            # Step 6
            results = {}
            M = {}
            for b in self.bundles:
                M[b] = self.create_EF(b=b, w=w_prev, x_bar=x_bar_prev, rho=self.rho)
                results[b] = solver.solve(M[b], solver_options=self.solver_options)
                #self.save_results(k, s, results[s], M[s])

            # Step 7
            x_bar = {}
            for x in self.first_stage_variables:
                x_bar[x] = 0.0
                for b in self.bundles:
                    x_bar[x] += data[b].probability * self.get_variable_value(x,M[b])

            # Step 8
            w = {}
            for b in self.bundles:
                w[b] = {}
                for x in self.first_stage_variables:
                    w[b][x] = w_prev[b][x] + self.rho * (self.get_variable_value(x,M[b]) - x_bar[x])

            # Step 9
            g = 0.0
            for b in self.bundles:
                g += data[b].probability * self.norm(self.get_variable_value(x,M[b]) - x_bar[x] for x in self.first_stage_variables)

            # Step 10
            if g < self.convergence_tolerance:
                break

            self.update_rho()

        self.store_results(x_bar=x_bar, w=w, g=g)
            
    def initialize_bundles(self, arg):
        # Setup self.bundles based on the args tne the user data
        # TODO HERE
        pass

    def update_rho(self):
        # TODO HERE
        pass

    def get_variable_value(self, v, M):
        # Abstract
        pass

    def create_EF(self, * b, w=None, x_bar=None, rho=None):
        # Abstract
        pass

    def store_results(self, *, x_bar, w, g):
        # Abstract
        pass
            
            
class ProgressiveHedgingSolve_Pyomo(ProgressiveHedgingSolver):

    def get_variable_value(self, v, M):
        pass
            

class Farmer(ProgressiveHedgingSolve_Pyomo):

    def __init__(self, data):
        self.data = data

    def initialize(self):
        self.first_stage_variables = []

    def create_EF(self, * b, w=None, x_bar=None, rho=None):
        pass

    def store_results(self, *, x_bar, w, g):
        pass
            
