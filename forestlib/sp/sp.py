import sys
import pprint
import json
import copy
import munch
import logging

from . import scentobund

#import forestlib.util
import forestlib.logs

logger = forestlib.logs.logger


class StochasticProgram(object):

    def __init__(self):
        self.solver = "gurobi"
        self._binary_or_integer_fsv = set()

        # Bundles (must be initialized later)
        self.bundles = None
        # Dictionary of application data
        self.app_data = {}
        # model_data[model_name] -> data
        self.model_data = {}
        # scenario_data[model_name][scenario_name] -> data
        self.scenario_data = {}

        # The name of the default model used to evaluate
        # solutions
        self.default_model = None

    def initialize_application(self, *, filename=None, app_data=None, **kwargs):
        if filename is not None:
            with open(f"{filename}", "r") as file:
                self.app_data = json.load(filename)
        elif app_data is not None:
            self.app_data = app_data

    def initialize_bundles(
        self,
        *,
        scheme=None,
        models=None,
        **kwargs,
    ):
        if scheme == None:
            scheme = "single_scenario"
        if models == None:
            models = [self.default_model] + list(
                sorted(
                    model
                    for model in self.scenario_data.keys()
                    if model != self.default_model
                )
            )
        else:
            for name in models:
                assert name in self.scenario_data

        assert len(models) > 0, "Cannot initialize bundles without model data"
        if "model_weight" in kwargs:
            model_weight = kwargs["model_weight"]
        else:
            model_weight = {
                model: mdata.get("_model_weight_", 1.0)
                for model, mdata in self.model_data.items()
            }
        self.bundles = scentobund.BundleObj(
            data=self.scenario_data,
            models=models,
            model_weight=model_weight,
            scheme=scheme,
            bundle_args=kwargs,
        )

    def get_bundles(self):
        if self.bundles is None:
            return None
        return munch.unmunchify(self.bundles._bundles)

    def save_bundles(self, json_filename, indent=None, sort_keys=False):
        self.bundles.dump(json_filename, indent=indent, sort_keys=sort_keys)

    def load_bundles(self, json_filename):
        self.bundles = scentobund.load_bundles(json_filename)

    def get_variables(self, b=None):
        if b is None:
            # If no value for 'b' is specified, then get the "first" bundle ID in self.bundles
            b = next(iter(self.bundles))

        # Return a dictionary mapping variable name to variable value, for all
        # first stage variables
        return {
            self.get_variable_name(b, v): self.get_variable_value(b, v)
            for v in self.shared_variables()
        }

    def get_variable_value(self, b, v):
        pass

    def get_variable_name(self, b, v):
        pass

    def fix_variable(self, b, v, value):
        pass

    def shared_variables(self):
        pass

    def set_solver(self, name):
        self.solver = name

    def solve(self, M, *, solver_options=None):
        pass

    def create_subproblem(self, b, *, w=None, x_bar=None, rho=None):
        return self.create_EF(b=b, w=w, x_bar=x_bar, rho=rho)

    def create_EF(self, *, b, w=None, x_bar=None, rho=None):
        pass

    def evaluate(self, x, solver_options=None):
        if solver_options is None:
            solver_options = {}

        # Setup single-scenario bundles with the default model
        _bundles = self.bundles
        self.initialize_bundles(models=[self.default_model], scheme="single_scenario")

        obj_value = {}
        M = {}
        for b in self.bundles:
            M[b] = self.create_subproblem(b)
            for i, xval in enumerate(x):
                self.fix_variable(b, i, xval)
            results = self.solve(M[b], solver_options=solver_options)
            if results.obj_value is None:
                msg = f"Error evaluating solution for scenario {b}\n\tVariables:\n\t\t"
                tmp = self.get_variables(b)
                msg = msg + "\n\t\t".join(
                    f"{var}:\t{tmp[var]}" for var in sorted(tmp.keys())
                )
                logger.debug(msg)
                return munch.Munch(feasible=False, bundle=b)
            else:
                obj_value[b] = results.obj_value
        obj = sum(self.bundles[b].probability * obj_value[b] for b in self.bundles)
        # Just need to get one of the bundles to collect the variables

        return munch.Munch(feasible=True, objective=obj, variables=self.get_variables())

        # Reset the bundles
        self.bundles = _bundles

