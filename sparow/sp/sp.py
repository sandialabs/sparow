import sys
import pprint
import json
import copy
import munch
import logging

from .bundling import bundling_functions

# import sparow.util
import sparow.logs

logger = sparow.logs.logger


def initialize_bundles(
    *,
    scheme=None,
    models=None,
    default_model=None,
    model_data=None,
    scenario_data=None,
    **kwargs,
):
    """
    Initialize bundles for stochastic programming.

    Parameters
    ----------
    scheme : str, optional
        The bundling scheme to use. Defaults to "single_scenario".
    models : list, optional
        List of model names to include in the bundles.
    default_model : str, optional
        The default model name.
    model_data : dict, optional
        Dictionary containing model data.
    scenario_data : dict, optional
        Dictionary containing scenario data.
    **kwargs : dict
        Additional keyword arguments for bundling.

    Returns
    -------
    BundleObj
        An initialized BundleObj instance.
    """
    if scenario_data is None:
        scenario_data = {}
    if model_data is None:
        model_data = {}

    if scheme == None:
        scheme = "single_scenario"
    if models == None:
        models = [default_model] + list(
            sorted(model for model in scenario_data.keys() if model != default_model)
        )
    else:
        for name in models:
            assert name in scenario_data

    assert len(models) > 0, "Cannot initialize bundles without model data"
    if "model_weight" in kwargs:
        model_weight = kwargs["model_weight"]
    else:
        model_weight = {
            model: mdata.get("_model_weight_", 1.0)
            for model, mdata in model_data.items()
        }

    if model_weight:
        return bundling_functions.BundleObj(
            data=scenario_data,
            models=models,
            model_weight=model_weight,
            scheme=scheme,
            bundle_args=kwargs,
        )
    else:
        return bundling_functions.BundleObj(
            data=scenario_data,
            models=models,
            scheme=scheme,
            bundle_args=kwargs,
        )


class StochasticProgram(object):
    """
    A class to represent a stochastic program.

    Attributes
    ----------
    solver : str
        The solver used for optimization (default is "gurobi").
    _binary_or_integer_fsv : set
        Set of binary or integer first-stage variables.
    bundles : BundleObj or None
        Bundles for the stochastic program.
    app_data : dict
        Dictionary of application data.
    model_data : dict
        Dictionary mapping model names to their data.
    scenario_data : dict
        Dictionary mapping model and scenario names to their data.
    default_model : str or None
        The name of the default model used to evaluate solutions.
    """

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
        """
        Initialize the application with data.

        Parameters
        ----------
        filename : str, optional
            Path to a JSON file containing application data.
        app_data : dict, optional
            Dictionary containing application data.
        **kwargs : dict
            Additional keyword arguments.
        """
        if filename is not None:
            with open(f"{filename}", "r") as file:
                self.app_data = json.load(filename)
        elif app_data is not None:
            self.app_data = app_data

    # DEPRECATED METHOD(?)
    def initialize_bundles(self, *, scheme=None, models=None, **kwargs):
        """
        Initialize bundles for the stochastic program.

        Parameters
        ----------
        scheme : str, optional
            The bundling scheme to use.
        models : list, optional
            List of model names to include in the bundles.
        **kwargs : dict
            Additional keyword arguments for bundling.
        """
        self.set_bundles(
            initialize_bundles(
                scheme=scheme,
                models=models,
                default_model=self.default_model,
                model_data=self.model_data,
                scenario_data=self.scenario_data,
                **kwargs,
            )
        )

    def set_bundles(self, bundles):
        """
        Set the bundles for the stochastic program.

        Parameters
        ----------
        bundles : BundleObj
            The bundles to set.
        """
        self.bundles = bundles

    def get_bundles(self):
        """
        Get the bundles for the stochastic program.

        Returns
        -------
        dict or None
            The bundles as a dictionary, or None if not initialized.
        """
        if self.bundles is None:
            return None
        return munch.unmunchify(self.bundles._bundles)

    def save_bundles(self, json_filename, indent=None, sort_keys=False):
        """
        Save the bundles to a JSON file.

        Parameters
        ----------
        json_filename : str
            Path to the JSON file where bundles will be saved.
        indent : int, optional
            Indentation level for JSON formatting.
        sort_keys : bool, optional
            Whether to sort keys in the JSON output.
        """
        self.bundles.dump(json_filename, indent=indent, sort_keys=sort_keys)

    def load_bundles(self, json_filename):
        """
        Load bundles from a JSON file.

        Parameters
        ----------
        json_filename : str
            Path to the JSON file containing bundles.
        """
        self.set_bundles(bundling_functions.load_bundles(json_filename))

    def get_variables(self, b=None):
        """
        Get variables for a specific bundle.

        Parameters
        ----------
        b : str, optional
            Bundle identifier. If None, the first bundle ID is used.

        Returns
        -------
        dict
            Dictionary mapping variable names to their values.
        """
        if b is None:
            # If no value for 'b' is specified, then get the "first" bundle ID in self.bundles
            b = next(iter(self.bundles))

        # Return a dictionary mapping variable name to variable value, for all
        # first stage variables
        return {
            self.get_variable_name(v): self.get_variable_value(b, v)
            for v in self.shared_variables()
        }

    def get_variable_value(self, b, v):
        """
        Get the value of a variable.

        Parameters
        ----------
        b : str
            Bundle identifier.
        v : object
            Variable object.

        Returns
        -------
        object
            The value of the variable.
        """
        pass

    def get_variable_name(self, v):
        """
        Get the name of a variable.

        Parameters
        ----------
        v : object
            Variable object.

        Returns
        -------
        str
            The name of the variable.
        """
        pass

    def fix_variable(self, b, v, value):
        """
        Fix a variable to a specific value.

        Parameters
        ----------
        b : str
            Bundle identifier.
        v : object
            Variable object.
        value : object
            Value to set for the variable.
        """
        pass

    def shared_variables(self):
        """
        Get the list of shared variables.

        Returns
        -------
        list
            List of shared variable objects.
        """
        pass

    def get_objective_coef(self, v):
        """
        Get the objective coefficient for a variable.

        Parameters
        ----------
        v : object
            Variable object.

        Returns
        -------
        float
            The objective coefficient for the variable.
        """
        pass

    def set_solver(self, name):
        """
        Set the solver for the stochastic program.

        Parameters
        ----------
        name : str
            Name of the solver to use.
        """
        self.solver = name

    def solve(self, M, *, solver_options=None):
        """
        Solve the stochastic program.

        Parameters
        ----------
        M : object
            Model to solve.
        solver_options : dict, optional
            Dictionary of solver options.

        Returns
        -------
        object
            Results of the optimization.
        """
        pass

    def create_EF(self, model_fidelities=None, cache_bundles=False):
        """
        Create the extensive form of the stochastic program.

        Parameters
        ----------
        model_fidelities : dict, optional
            Dictionary specifying model fidelities.
        cache_bundles : bool, optional
            Whether to cache bundles (default is False).

        Returns
        -------
        object
            The extensive form model.
        """
        pass

    def create_subproblem(
        self, b, *, w=None, x_bar=None, rho=None, cached=False, compact_repn=True
    ):
        """
        Create a subproblem for a specific bundle.

        Parameters
        ----------
        b : str
            Bundle identifier.
        w : object, optional
            Weight vector.
        x_bar : object, optional
            Fixed variable values.
        rho : object, optional
            Penalty parameter.
        cached : bool, optional
            Whether to use cached bundles (default is False).
        compact_repn : bool, optional
            Whether to use compact representation (default is True).

        Returns
        -------
        object
            The subproblem model.
        """
        return self.create_bundle_EF(
            b=b, w=w, x_bar=x_bar, rho=rho, cached=cached, compact_repn=compact_repn
        )

    def create_bundle_EF(
        self, *, b, w=None, x_bar=None, rho=None, cached=False, compact_repn=True
    ):
        """
        Create the extensive form for a specific bundle.

        Parameters
        ----------
        b : str
            Bundle identifier.
        w : object, optional
            Weight vector.
        x_bar : object, optional
            Fixed variable values.
        rho : object, optional
            Penalty parameter.
        cached : bool, optional
            Whether to use cached bundles (default is False).
        compact_repn : bool, optional
            Whether to use compact representation (default is True).

        Returns
        -------
        object
            The extensive form model for the bundle.
        """
        pass

    def evaluate(self, x, solver_options=None, cached=False):
        """
        Evaluate a solution for the stochastic program.

        Parameters
        ----------
        x : list
            List of variable values to evaluate.
        solver_options : dict, optional
            Dictionary of solver options.
        cached : bool, optional
            Whether to use cached bundles (default is False).

        Returns
        -------
        Munch
            A Munch object containing evaluation results with keys:
            - feasible (bool): Whether the solution is feasible.
            - objective (float): The objective value.
            - variables (dict): Dictionary of variable values.
        """
        if solver_options is None:
            solver_options = {}

        # Setup single-scenario bundles with the default model
        _bundles = self.bundles
        self.initialize_bundles(models=[self.default_model], scheme="single_scenario")

        obj_value = {}
        M = {}
        for b in self.bundles:
            M[b] = self.create_subproblem(b, cached=cached)
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

        retval = munch.Munch(
            feasible=True, objective=obj, variables=self.get_variables()
        )

        # Reset the bundles
        self.set_bundles(_bundles)

        return retval
