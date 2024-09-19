
class ProgressiveHedgingSolver(object):

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
            
            
class ProgressiveHedgingSolver_Pyomo(ProgressiveHedgingSolver):

    def __init__(self, first_stage_variables):
        #
        # A list of string names of variables, such as:
        #   [ "x", "b.y", "b[*].z[*,*]" ]
        #
        self.first_stage_variables = first_stage_variables
        self.varcuid_to_int = []

    def initialize_varmap(self, b, M):
        if len(self.varcuid_to_int) == 0:
            #
            # self.varcuid_to_int maps the cuids for variables to unique integers (starting with 0).
            #   The variable cuids indexed here are specified by the list self.first_stage_variables.
            #
            self.varcuid_to_int = []
            #for var in M.component_map(Var).values():
            for varname in first_stage_variables:
                cuid = ComponentUID(varname)
                comp = cuid.find_component_on(M)
                assert comp is not None, "Pyomo error: Unknown variable '%s'" % varname
                if comp.ctype is IndexedComponent:
                    for var in comp.values():    
                        self.varcuid_to_int[ pyo.ComponentUID(var) ] = len(self.varcuid_to_int)
                elif comp.ctype is Var:
                    self.varcuid_to_int[ pyo.ComponentUID(comp) ] = len(self.varcuid_to_int)
                else:
                    raise RuntimeError("Pyomo error: Component '%s' is not a variable" % varname)

        #
        # self.int_to_var maps the tuple (b,vid) to a Pyomo Var object, where b is a bundle ID and vid
        #   is an integer variable id (0..N-1) that is associated with the variables specified in
        #   self.first_stage_variables.
        #
        for cuid,vid in self.varcuid_to_int.items():
            self.int_to_var[b, vid] = cuid.find_component_on(M)

    def get_variable_value(self, b, v):
        return pyo.value(self.int_to_var[b, v])
            

class Farmer(ProgressiveHedgingSolver_Pyomo):

    def __init__(self, data):
        self.data = data

    def initialize(self):
        self.initialize_varmap()
        self.first_stage_variables = []

    def create_EF(self, * b, w=None, x_bar=None, rho=None):
        pass

    def store_results(self, *, x_bar, w, g):
        pass
            
