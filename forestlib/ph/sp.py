import pyomo.core.base.indexed_component
import pyomo.environ as pyo


class StochasticProgram(object):

    def __init__(self):
        self.solver = "gurobi"
        self.bundles = []

    def initialize_bundles(self, arg):
        # Setup self.bundles based on the args tne the user data
        # TODO HERE
        pass

    def bundle_probability(self, index):
        pass

    def initialize_varmap(self, *, b, M):
        pass

    def get_variable_value(self, v, M):
        pass

    def shared_variables(self):
        pass

    def solve(self, M, *, solver_options=None):
        pass

    def create_subproblem(self, * b, w=None, x_bar=None, rho=None):
        M = self.create_EF(b=b, w=w, x_bar=x_bar, rho=rho)
        self.initialize_varmap(b=b, M=M)
        return M

    def create_EF(self, * b, w=None, x_bar=None, rho=None):
        pass

            
class StochasticProgram_Pyomo(StochasticProgram):

    def __init__(self, *, first_stage_variables):
        StochasticProgram.__init__(self)
        #
        # A list of string names of variables, such as:
        #   [ "x", "b.y", "b[*].z[*,*]" ]
        #
        self.first_stage_variables = first_stage_variables
        self.varcuid_to_int = []
        self.int_to_var = {}
        self.solver_options = {}
        self.pyo_solver = None

    def initialize_varmap(self, *, b, M):
        if len(self.varcuid_to_int) == 0:
            #
            # self.varcuid_to_int maps the cuids for variables to unique integers (starting with 0).
            #   The variable cuids indexed here are specified by the list self.first_stage_variables.
            #
            self.varcuid_to_int = {}
            for varname in self.first_stage_variables:
                cuid = pyo.ComponentUID(varname)
                comp = cuid.find_component_on(M)
                assert comp is not None, "Pyomo error: Unknown variable '%s'" % varname
                if comp.ctype is pyomo.core.base.indexed_component.IndexedComponent:
                    for var in comp.values():    
                        self.varcuid_to_int[ pyo.ComponentUID(var) ] = len(self.varcuid_to_int)
                elif comp.ctype is pyo.Var:
                    self.varcuid_to_int[ pyo.ComponentUID(comp) ] = len(self.varcuid_to_int)
                else:
                    raise RuntimeError("Pyomo error: Component '%s' is not a variable" % varname)

        #
        # self.int_to_var maps the tuple (b,vid) to a Pyomo Var object, where b is a bundle ID and vid
        #   is an integer variable id (0..N-1) that is associated with the variables specified in
        #   self.first_stage_variables.
        #
        self.int_to_var[b] = {}
        for cuid,vid in self.varcuid_to_int.items():
            self.int_to_var[b][vid] = cuid.find_component_on(M)

    def get_variable_value(self, b, v):
        return pyo.value(self.int_to_var[b][v])
            
    def shared_variables(self):
        return list(range(len(self.varcuid_to_int)))

    def solve(self, M, *, solver_options=None, tee=False):
        if solver_options:
            self.solver_options = solver_options
        if self.pyo_solver is None:
            self.pyo_solver = pyo.SolverFactor(self.solver)

        res = self.pyo_solver.solve(M, solver_options=self.solver_options, tee=tee)
        return res
