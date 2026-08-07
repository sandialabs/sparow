import sys
import numpy as np
import munch
import logging
import datetime
import copy
import pprint

# general pyomo imports
from pyomo.common.timing import tic, toc, TicTocTimer
import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.repn

# sparow imports
from sparow import solnpool
import sparow.logs
from sparow.sp.bundling import bundling_functions

# or-topas imports
import or_topas.benders
from or_topas.benders.benders_serial import BendersGenerator_Serial
from or_topas.util.pyomo_utils import split_expr

# logger settup
logger = sparow.logs.logger


class BendersSolver(object):
    """
    This class provides an way to solve Stochastic Programs (or other SP formatted problems)
    using Benders Decomposition. It relies on the OR-TOPAS Benders solver for its funcitonality.

    The following structure is assumed for what the extensive form (or deterministic equivalent) problem
    looks like:
    min_{x,y_s} <c,x> + sum_{s \in S} p_s<d_s,y_s>
        s.t.    x in X subseteq Z^{n_1} cross R^{n_2}
                y_s in Y_s(x) subseteq R^{m_s}, forall s in S

    N.B. x are first-stage variables and y are second-stage variables.
    S is the scenaro set and p in R_+^{|S|}.

    The assumed formats for X and Y_s are as follows:
    X = {x in Z^{n_1} cross R^{n_2} | Ux <= v}
    Y_s = {y_s in R^{m_s} | A_s x  + B_s y_s <= h_s, E_s x<= g_s}

    The Benders Master problem then takes the form:
    min_{x,eta} <c,x> + sum_{s \in S} eta_s
        s.t.    x in X subseteq Z^{n_1} cross R^{n_2}
                eta_s >= f_{c,s}(x), forall c in OptCuts_s, s in S
                0 >= f_{c,s}(x), forall c in FeasCuts_s, s in S

    The Benders Subproblem Value Functions then take the form (s index dropped for convience):
    Q_s(bar{x}) := min_{x,y} p_s[<c,x> + <d,y>]
                     s.t.    Ax + By <= h,
                             Ex      <= g,
                             x       == bar{x},
                             x in R^{n}, n = n_1 + n_2
                             y_s in R^{m}

    !!!!!!!!!!
    Warning:
    1. The probabiilty weights are pushed to the subproblem value functions.
    This allows the master problem to not need to know the probability vector.

    2. Multiple SP objects are created for this process: sp_lower and sp_upper.
    sp_lower has the full scenario information the user inputted.
    sp_upper is created with a single scenario from the information given by the user.
    This separates the handling of the |S|+1 models into conceptually separated objects.

    3. The bundles in sp_lower will be converted in place into Benders subproblem format.

    4. The single bundle in sp_upper will be converted in place into the Benders master problem.

    5. All the in-place model transformation into Benders format is a conceptual difference from how the rest of
    Sparow works. It is necessary for Benders Decomposition and the use of OR-TOPAS for Benders Cut Generation.
    This means that the models used for subproblems are converted into Benders subproblems.
    The models used for master problems are converted into master problems.
    This is not a tranform that can directly easily reverse or deactivated.
    This holds true even though the models are attached to an sparow sp object.
    !!!!!!!!!!!
    """

    #
    # Init and set_options adapted from ph.py
    #
    def __init__(self):
        self.max_iterations = 100
        self.convergence_tolerance = 1e-3
        self.solver_name = None
        self.subproblem_solver_name = None
        self.solver_options = {}
        self.relax_subproblem_integrality = False
        self.support_feasibility_cuts = False
        self.solutions = None
        self.master_problem_bundle = None
        self.BendersCutGenerator = BendersGenerator_Serial
        self.BendersTransform = "standard_lp"
        self.is_persistent_solver = None
        self.custom_b_upper = None

    def set_options(
        self,
        *,
        max_iterations=None,
        convergence_tolerance=None,
        solver=None,  # keeping this for parallelism with PH, attribute is actually solver_name
        subproblem_solver=None,
        solver_options=None,
        master_problem_bundle=None,
        relax_subproblem_integrality=None,
        support_feasibility_cuts=None,
        solutions=None,
        loglevel=None,
        BendersCutGenerator=None,
        is_persistent_solver=False,
        allow_infeasible_subproblems=False,
        custom_b_upper=None,
    ):

        assert solver is not None, "Need to declare an upper level solver"
        assert subproblem_solver is not None, "Need to declare a subproblem solver"
        self.solver_name = solver
        self.subproblem_solver_name = subproblem_solver

        #
        # Misc configuration
        #
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if convergence_tolerance is not None:
            self.convergence_tolerance = convergence_tolerance
        if solver_options is not None:
            self.solver_options = solver_options
        if solutions is not None:
            self.solutions = solutions
        if master_problem_bundle is not None:
            self.master_problem_bundle = master_problem_bundle
        if relax_subproblem_integrality is not None:
            self.relax_subproblem_integrality = relax_subproblem_integrality
        if support_feasibility_cuts is not None:
            self.support_feasibility_cuts = support_feasibility_cuts
        if BendersCutGenerator is not None:
            self.BendersCutGenerator = BendersCutGenerator
        if is_persistent_solver is not None:
            self.is_persistent_solver = is_persistent_solver
        if allow_infeasible_subproblems is not None:
            self.allow_infeasible_subproblems = allow_infeasible_subproblems
        if custom_b_upper is not None:
            self.custom_b_upper = custom_b_upper

        if loglevel is not None:
            if loglevel == "DEBUG" or loglevel == "VERBOSE":
                sparow.logs.use_debugging_formatter()
            logger.setLevel(loglevel)

    @staticmethod
    def _transform_to_subproblem_model(
        sp_lower,
        b,
        default_domain=pyo.Reals,
        remove_first_stage_only_cons=False,
        weight_obj_by_prob=True,
        remove_first_stage_objective_terms=False,
        additional_transforms=None,
    ):
        """
        This method takes a sp object and one of its child single scenario models (or bundle), b,
        and converts them to the corresponding Benders subproblem.
        The modification is done to the model inplace.
        N.B. x are first-stage variables and y are second-stage variables.

        Scenario specific data is indexed with s and probability p_s
        It assumes problems of the following form:
        min_{x,y} <c,x> + <d,y>
        s.t.    Ax + By <= h,
                Ex      <= g,
                x in Z^{n_1} cross R^{n_2},
                y_s in R^{m}

        It converts to a problem of the form:
        min_{x,y} p_s[<c,x> + <d,y>]
            s.t.    Ax + By <= h,
                    Ex      <= g,
                    x       == bar{x},
                    x in R^{n}, n = n_1 + n_2
                    y_s in R^{m}

        N.B. If remove_first_stage_objective_terms == True, the <c,x> term in the objective is removed.
        It will not remove something like a tracking variable \theta from an objective of <c,x> + <d,y> + \theta, where \theta >= <p,x>.

        This is done by achieving several steps:

        1. updating the objective
            1a. if remove_first_stage_objective_terms == True, remove c^Tx terms.
            1b. Weight the objective by probability p_s.
        2. relaxing discrete first-stage variables to continuous,
        3. optionally remove the first-stage only constraints (these may be scenario specific feasbility constraints)
        4. remove any exteraneous PH style attributes

        There are optional steps that may be added later:
        5. relaxing second-stage variables as well
        6. ability to set custom domains as a function of which first stage variable it is


        N.B. that this method on its own does not convert into a value function.
        That is done later with OR-TOPAS, where the problem will be modified to make:
        Q_s(bar{x}) := min_{x,y} p_s[<c,x> + <d,y>]
                     s.t.    Ax + By <= h,
                             Ex      <= g,
                             x       == bar{x},
                             x in R^{n}, n = n_1 + n_2
                             y_s in R^{m}
        So the x == bar{x} functionality is done by OR-TOPAS's benders solver
        This structure also has as a default that the <c,x> terms are left on in the subproblem.
        We default to this here because it is an expensive step if it can be avoided.
        Control of remove_first_stage_objective_terms is passed through to the solve method, which defaults to
        remove_first_stage_objective_terms = True in solve at present.
        """
        assert (
            len(sp_lower.bundles[b].scenarios) == 1
        ), "There should be only one scenario in this bundle"

        subproblem_model = sp_lower.create_bundle_EF(
            b=b,
            w=None,
            x_bar=None,
            rho=None,
            cached=False,
            compact_repn=True,
        )
        # we now assume subproblem_model is constructed (and the associated cuid map initialized)

        subproblem_first_stage_vars = None

        # step 1: updating the objective
        # Note: we assume the objective is called obj
        if not hasattr(subproblem_model, "obj") or not subproblem_model.obj.active:
            raise ValueError(f"No active objective found on subproblem for bundle {b}")
        original_expr = subproblem_model.obj.expr
        original_sense = subproblem_model.obj.sense
        subproblem_model.obj.deactivate()
        expr_holder = original_expr

        # Handle the removal of <c,x> terms if desired.
        if remove_first_stage_objective_terms:
            # This implicitly enforces a linearity assumption on the objective.
            # The linearity check is buried in how or_topas.util.pyomo_utils.split_expr
            #   will handle parsing using Pyomo's get_standard_repn.

            # N.B. this subproblem_first_stage_vars is a set coming from a dict .values() method
            subproblem_first_stage_vars = ComponentSet(
                sp_lower.int_to_FirstStageVar[b].values()
            )
            obj_split = split_expr(
                expr_holder, subproblem_first_stage_vars, allow_iterables=True
            )
            expr_holder = obj_split.not_in_set + obj_split.constant
        # Handle probablity weighting
        if weight_obj_by_prob:
            # we use the sp.bundles[b].probability prob here
            # we want to use the version that ripples to thetas in master here.
            p_s = sp_lower.bundles[b].probability
            expr_holder = p_s * expr_holder

        # need to update the objective in place
        subproblem_model.obj = pyo.Objective(expr=expr_holder, sense=original_sense)

        # step 2: relax first_stage variable domains
        for i, x in sp_lower.int_to_FirstStageVar[b].items():
            x.domain = default_domain

        # step 3:
        if remove_first_stage_only_cons:
            # since this is a Benders subproblem, we can remove any constriant that
            # only involves first-stage variables.
            # This is equivalent to removing those that do not invovle second-stage variables.
            cons_to_delete = []
            if subproblem_first_stage_vars is None:
                subproblem_first_stage_vars = sp_lower.int_to_FirstStageVar[b].values()
            for cons in subproblem_model.component_data_objects(
                pyo.Constraint, descend_into=True
            ):
                if all(
                    var in subproblem_first_stage_vars
                    for var in pyo.visitor.identify_variables(cons.body)
                ):
                    # this amounts to an if all vars in first_stage_vars check
                    # sp.varcuid_to_int only holds varcuid's for first_stage_variables
                    cons_to_delete.append(cons)
            # unneed cons handling point
            for cons in cons_to_delete:
                # we can either deactivate or delete
                # deactivating for now
                cons.deactivate()
        # step 4: remove any unneeded PH style variable fluff
        # this may be all handled by setting w, x_bar, rho, and cached to false

        # step 5, apply transforms if given
        if additional_transforms:
            for tf in additional_transforms:
                result = tf(sp_lower, subproblem_model)
                if result is not None:
                    subproblem_model = result

        return subproblem_model

    @staticmethod
    def _create_sp_upper(sp_lower):
        """
        This method takes an sp object and a bundle and uses that to make a separate sp object with data
        for just that one bundle b.
        """
        return BendersSolver._create_sp_upper_large_copy(sp_lower)

    @staticmethod
    def _create_sp_upper_large_copy(sp_lower):
        # TODO: this is a possibly quite large deepcopy.
        # look at more efficient ways to accomplish the goal of creating an sp object for master problem data
        sp_upper = copy.deepcopy(sp_lower)
        return sp_upper

    @staticmethod
    def _transform_to_master_model(
        *,
        sp,
        b,
        eta_bounds_map,
        lower_bounding_otherwise_enforced=False,
        fix_second_stage_vars=False,
        objective_sense=pyo.minimize,
        etas_ordered=False,
        additional_transforms=None,
    ):
        """
        This method takes a sp object and one of its child single scenario models, b,
        and converts it to the format of a Benders master problem.
        The modification is done to the model inplace.
        N.B. x are first-stage variables and y are second-stage variables.
        Scenarios are assumed to be numbered 1 to n.

        It assumes problems of the following form:
        min_{x,y} <c,x> + <d,y>
        s.t.    Ax + By <= h,
                Ex      <= g,
                x in Z^{n_1} cross R^{n_2},
                y_s in R^{m}

        It converts to a problem of the form:
        min_{x,eta} <c,x> + sum_{i=1...n} eta_i
            s.t.    Ex      <= g,
                    x in Z^{n_1} cross R^{n_2},
                    eta_i in [LB_i, UB_i]

        This is done by taking several steps:
        1. Removal of second-stage terms from objective
        2. Deletion of all constraints that contain second-stage variables (more generally non-first-stage variables).
        3. Creating subproblem tracking variables, eta, for each scenario with domain bounds
        4. Adding sum of tracking variables, eta, to the objective

        !!!!!!!!!!!!!!!!!!!!!
        Note: this model is expected to function as the master problem.
        It assumes that the general first-stage constraints X subseteq G_s := {x in Z^{n_1} cross R^{n_2} | E_s x <= g_s},
            Hence using x in G_s is effectively pulling feasibility cut information from subproblem s to the master.
        And that c is the first-stage linear objective (i.e. c is not scenario specific), and that the objective is entirely affine.

        If this is not the case generally, the user must guarantee the given scenario/bundle has this property.
        This code is run after _create_sp_upper constructs an appropriate sp_upper.

        This is also an expensive transform, it should only be run once per solve to create the initial master model.
        !!!!!!!!!!!!!!!!!!!!!

        Parameters
        ----------
        sp : StochasticProgram
            The StochasticProgram object containing the model to convert to the Benders Master problem
        b : BundleObj
            The BundleObj object (custom to this library).
        eta_bounds_map : dictionary or ComponentMap
            A dicitonary mapping bundle keys to bound tuples.
            The tuples are in (LowerBound, UpperBound) format.
            The keys are assumed to be bundle ids corresponding to Benders subproblems.
            Constant scenarios should use (0,0) as bound format.
        lower_bounding_otherwise_enforced : Boolean
            A boolean stating if lower bounding of the master objective will be handled external to this code
            This allows etas to not need a non-None lower bound term in the eta_bounds_map tuples.
        fix_second_stage_vars : Boolean
            A boolean controlling if second-stage variables in this model are fixed to zero of left as free.
            Either way, the constraints that use them are deactivated.
            This can change the behavior of presolvers and alternative solution generation codes.
        objective_sense : pyomo.environ.minimize | pyomo.environ.maximize
            A pyomo.environ.sense option to select if the objective should be minimization or maximization.
            Default is minimization.
        etas_ordered : Boolean
            A boolean flag that lets the user control if the scenario set created from etas is ordered
        additional_transforms : None or list[Functions]
            An optional argument to apply transform functions to the model.
            Passed Transforms must accept a stochastic program and pyomo model as arguments
        """
        assert objective_sense in (
            pyo.minimize,
            pyo.maximize,
        ), "Sense needs to be a valid sense"
        assert (
            len(sp.bundles[b].scenarios) == 1
        ), "There should be only one scenario in this bundle"
        assert (
            len(eta_bounds_map) > 0
        ), "Need there to be at least some scenarios to map to"
        for k, v in eta_bounds_map.items():
            assert (
                isinstance(v, tuple) and len(v) == 2
            ), f"Tried to use {v=} as a bound tuple"
            if not lower_bounding_otherwise_enforced:
                assert v[0] is not None, f"Subproblem {k=} must have a lower bound"

        # step 0: create master problem model
        # see discussion in _transform_to_subproblem_model about best method to use here
        upper_model = sp.create_bundle_EF(
            b=b,
            cached=False,
            w=None,
            x_bar=None,
            rho=None,
            compact_repn=True,
        )

        # we will use this component set for several inclusion checks later on.
        first_stage_vars = ComponentSet(sp.int_to_FirstStageVar[b].values())

        # N.B. since this is a single scenario model, it's creation logic will rely on _create_single_scenario_EF_repn
        # the information from that model is then put in EF by ConcreteModel's transfer_attributes_from method
        # so the cuids and maps will be based on this correct model.

        # step 1: compute the objective with just the first stage terms plus constant
        # name the resulting expression the first_stage_cost
        # note the assumption here that the objective is affine
        # this is intentionally done before any of the other alterantions to the model
        repn = pyomo.repn.generate_standard_repn(upper_model.obj.expr, quadratic=False)
        first_stage_cost = repn.constant if repn.constant is not None else 0
        for i, var in enumerate(repn.linear_vars):
            if var in first_stage_vars:
                # this is the 'is first stage variable' check
                # assumes that all vars in repn.linear_vars come from just the present upper_model
                first_stage_cost += repn.linear_coefs[i] * var

        # step 2: delete all constraints involving second-stage variables and second-stage variables
        # this removes all the recourse/scenario specific feasibility constraints/variables

        # gather cons to remove to process later
        # TODO: consider caching the CUIDs for second-stage vars
        # TODO: do we need to delete inactive constraints
        cons_to_delete = []
        for cons in upper_model.component_data_objects(
            pyo.Constraint, descend_into=True
        ):
            # variable listing method adapted from
            # https://stackoverflow.com/questions/48538945/access-all-variables-occurring-in-a-pyomo-constraint

            if any(
                var not in first_stage_vars
                for var in pyo.visitor.identify_variables(cons.body)
            ):
                # this amounts to an any vars not in first_stage_vars check
                # sp.varcuid_to_int only holds varcuid's for first_stage_variables
                cons_to_delete.append(cons)

        # unneed cons handling point
        for cons in cons_to_delete:
            # we can either deactivate or delete
            # deactivating for now
            cons.deactivate()

        # unneeded vars handling point
        if fix_second_stage_vars:
            # gather vars to remove to process later
            vars_to_delete = []
            for var in upper_model.component_data_objects(pyo.Var, descend_into=True):
                if var not in first_stage_vars:
                    vars_to_delete.append(var)

            for var in vars_to_delete:
                # we can either fix or delete
                # fixing for now because we do not want unneed slack variables when generating alternative solutions
                var.domain = pyo.Reals
                var.fix(0)

        # step 3: create eta tracking variables for the scenario set (either 1...N or a pyomo set)
        # enforce that there are upper and lower bound entries in the eta_bounds dict for each scenario
        # none corresponds to no bound on that side
        # feasibility only problems can be set to have eta_i = 0 by LB_i = UB_i = 0
        upper_model.scenarios = pyo.Set(
            initialize=list(eta_bounds_map.keys()), ordered=etas_ordered
        )

        def eta_bound_rule(m, s):
            return eta_bounds_map[s]

        upper_model.etas = pyo.Var(upper_model.scenarios, bounds=eta_bound_rule)

        # step 4: add eta.sum to the objective expression
        if (
            hasattr(upper_model, "obj")
            and hasattr(upper_model.obj, "active")
            and upper_model.obj.active
        ):
            upper_model.obj.deactivate()
        upper_model.obj = pyo.Objective(
            expr=first_stage_cost
            + sum(upper_model.etas[s] for s in upper_model.scenarios),
            sense=objective_sense,
        )

        # step 5, apply transforms if given
        if additional_transforms:
            for tf in additional_transforms:
                result = tf(sp, upper_model)
                if result is not None:
                    upper_model = result

        return upper_model

    @staticmethod
    def _setup_topas_subproblem(
        sp_lower,
        b_lower,
        sp_upper,
        b_upper,
        remove_first_stage_objective_terms,
        additional_transforms,
    ):
        """
        This is a wrapper method that takes a stochastic program and bundle, sp_lower and b_lower,
        and returns data for use as an OR-TOPAS Benders subproblem.

        At present, it builds the needed map from the first-stage variables in m_upper to those in b.
        It then returns the subproblem model, b, and that complicating variable map.
        """
        # rely on _transform_to_subproblem_model to create the subproblem model
        model_lower = BendersSolver._transform_to_subproblem_model(
            sp_lower,
            b_lower,
            default_domain=pyo.Reals,
            remove_first_stage_objective_terms=remove_first_stage_objective_terms,
            additional_transforms=additional_transforms,
        )

        # create the complicating variable map
        # use the sp objects both having first stage variable lists to build the mapping

        # so self.int_to_FirstStageVarName is not indexed by bundles, so for compact_repn models
        # it should be identical between different bundles (and objects created from the same data)
        # it also means that the component map should be
        # this format actually does not require that sp_upper and sp_lower are separate sp objects, but it can handle separate sp objects

        complicating_variable_map = ComponentMap()

        # in the case where there is one first stage variable 'x'
        # this should link upper_model.s[None, b_upper].x to lower_model.s[None, b_lower].x
        for i, var_upper in sp_upper.int_to_FirstStageVar[b_upper].items():
            complicating_variable_map[var_upper] = sp_lower.int_to_FirstStageVar[
                b_lower
            ][i]

        if logger.isEnabledFor(logging.DEBUG):
            print(f"Subproblem {b_lower=} Initial Setup Start Pretty Print:")
            model_lower.pprint()
            print(f"Subproblem {b_lower=} Initial Setup End Pretty Print")
        # return the subproblem model, b, and the complicating variable map.
        return model_lower, complicating_variable_map

    def solve_and_return_model(
        self,
        sp_lower,
        eta_bounds_map,
        error_on_initialized_root_vars=False,
        convergence_tol=1e-8,
        subproblem_transforms=None,
        master_transforms=None,
        on_iteration=None,
        **options,
    ):
        # steps for general solve
        # take overall sp model with full scenario data, this will be sp_lower
        # create sp model with single scenario for sp_upper
        # for each bundle in sp_lower, transform to subproblem
        # transform sp_upper model to master format
        # setup or-topas benders model
        # extract subproblem solver information from sp_lower
        # iterate while adding benders cuts
        # report results to user

        # TODO: at present we are saying the master and subproblem solver must have the same
        # persistence behavior, we could move to tracking is_persistent automatically based off solver name
        # in future

        start_time = datetime.datetime.now()
        if eta_bounds_map is None:
            raise AssertionError(
                f"eta_bounds_map was None, expect something like dict(s: (default_lower_eta, None) for s in sp_lower.bundles) where sp.bundles has structure {len(sp_lower.bundles)=} and {sp_lower.bundles.keys()=}"
            )
        assert eta_bounds_map is not None, "Must give a valid bounds map"
        assert (
            self.subproblem_solver_name is not None
        ), "Must give a valid subproblem solver name"
        if self.allow_infeasible_subproblems:
            assert (
                self.is_persistent_solver
            ), "Must use a persistent solver to support feasibility cuts"

        if len(options) > 0:
            self.set_options(**options)
        if logger.isEnabledFor(logging.DEBUG):
            print("Solver Configuration")
            print(f"  max_iterations               {self.max_iterations}")
            print(f"  solver_name                  {self.solver_name}")
            print(f"  subproblem_solver_name       {self.subproblem_solver_name}")
            print(f"  relax_subproblem_integrality {self.relax_subproblem_integrality}")
            print(f"  support_feasibility_cuts     {self.support_feasibility_cuts}")
            print(f"  is_persistent_solver         {self.is_persistent_solver}")
            print(f"  allow_infeasible_subproblems {self.allow_infeasible_subproblems}")
            print(f"  custom_b_upper               {self.custom_b_upper}")
            print("")

        #
        # Setup solution manager and archive context information
        #
        if self.solutions is None:
            self.solutions = solnpool.SparowPoolManager()

        # consider making collection of iterates optional
        sp_metadata = self.solutions.add_pool(
            name="Benders Iterations", policy=solnpool.PoolPolicy.keep_latest
        )
        sp_metadata.solver = "Benders Iteration Results"
        # TODO: update info cached here
        sp_metadata.solver_options = dict(
            cached_model_generation=False,
            max_iterations=self.max_iterations,
            convergence_tolerance=self.convergence_tolerance,
            # normalize_convergence_norm=self.normalize_convergence_norm,
            solver_name=self.solver_name,
            solver_options=self.solver_options,
        )

        # TODO: add check that the single scenario to bundle rules are followed
        # we may be able to relax this later

        logger.info("BendersSolver - START")
        iteration_timer = TicTocTimer()
        iteration_timer.tic(None)

        # create master problem
        tic("Creating Benders Master Problem", logger=logger, level=logging.VERBOSE)
        sp_upper = BendersSolver._create_sp_upper(sp_lower=sp_lower)

        if self.custom_b_upper is not None:
            b_upper = self.custom_b_upper
        else:
            b_upper = next(iter(sp_upper.bundles))

        # TODO: update all of these to be parameters for the solver later
        upper_model = BendersSolver._transform_to_master_model(
            sp=sp_upper,
            b=b_upper,
            eta_bounds_map=eta_bounds_map,
            lower_bounding_otherwise_enforced=False,
            fix_second_stage_vars=True,
            objective_sense=pyo.minimize,
            etas_ordered=False,
            additional_transforms=master_transforms,
        )
        # upper_model.s[None,'High'].x[0].fix(1)
        # upper_model.s[None,'High'].x[1].fix(0)
        # upper_model.s[None,'High'].x[2].fix(1)
        if logger.isEnabledFor(logging.DEBUG):
            print(f"Upper Model Initial Setup Start Pretty Print:")
            upper_model.pprint()
            print(f"Upper Model Initial Setup End Pretty Print:")

        root_vars = list(sp_upper.int_to_FirstStageVar[b_upper].values())
        unitialized_root_vars = [rv for rv in root_vars if rv.value is None]
        if len(unitialized_root_vars) > 0:
            if error_on_initialized_root_vars:
                raise RuntimeWarning(f"There are root variables without initial values")
            for rv in unitialized_root_vars:
                if rv.lb is not None:
                    rv.value = rv.lb
                elif rv.ub is not None:
                    rv.value = rv.ub
                else:
                    rv.value = 0
            if logger.isEnabledFor(logging.DEBUG):
                print(f"Gave default values to {len(unitialized_root_vars)} variables")

        # create topas Benders object

        upper_model.benders = self.BendersCutGenerator()
        # TODO: add the allow feasibility cut flags here
        upper_model.benders.set_input(
            root_vars=root_vars,
            tol=convergence_tol,
            transform=self.BendersTransform,
            allow_infeasible=self.allow_infeasible_subproblems,
        )
        # create master solver object
        # if self.is_persistent_solver:
        #     raise RuntimeError(f"Not Supporting Peristent Solvers at present")
        # else:
        #     opt = pyo.SolverFactory(self.solver_name)
        opt = pyo.SolverFactory(self.solver_name)
        if self.is_persistent_solver:
            opt.set_instance(upper_model)
        # TODO: handle persistent solver setup

        # add subproblems
        tic("Creating subproblems", logger=logger, level=logging.VERBOSE)
        for b_lower in sp_lower.bundles:
            subproblem_fn_kwargs = dict()
            subproblem_fn_kwargs["sp_lower"] = sp_lower
            subproblem_fn_kwargs["b_lower"] = b_lower
            subproblem_fn_kwargs["sp_upper"] = sp_upper
            subproblem_fn_kwargs["b_upper"] = b_upper
            subproblem_fn_kwargs["remove_first_stage_objective_terms"] = True
            subproblem_fn_kwargs["additional_transforms"] = subproblem_transforms
            if logger.isEnabledFor(logging.DEBUG):
                print(f"Subproblem {b_lower=} debug info")
                print(f"Subproblem setup keywords:")
                print(subproblem_fn_kwargs)
            upper_model.benders.add_subproblem(
                subproblem_fn=BendersSolver._setup_topas_subproblem,
                subproblem_fn_kwargs=subproblem_fn_kwargs,
                root_eta=upper_model.etas[b_lower],  # this may be finicky
                subproblem_solver=self.subproblem_solver_name,  # make sure this is initialized above
            )

        #
        iteration = 0
        termination_condition = "Termination: unknown"
        while True:
            iteration_timer.tic(None)
            iteration += 1

            # possibly add a toc for this iteration start
            # time_last_iter = iteration_timer.toc(None)

            # possibly add a log iteration here.

            # add Benders iteration here
            # handle non-persistent case
            if self.is_persistent_solver:
                # raise RuntimeError(f"Not Supporting Persitent Solvers at present")
                res = opt.solve(tee=False, save_results=False)
                cuts_added = upper_model.benders.generate_cut()
                for c in cuts_added:
                    opt.add_constraint(c)
            else:
                res = opt.solve(
                    upper_model,
                    tee=False,
                )
                cuts_added = upper_model.benders.generate_cut()

            if on_iteration:
                on_iteration(
                    data=munch.Munch(
                        iter_idx=iteration,
                        cuts_added=cuts_added,
                        upper_model=upper_model,
                    )
                )

            if len(cuts_added) == 0:
                termination_condition = f"Termination: No Cuts Added"
                logger.info(termination_condition)
                break
                # add no cuts added break here

            # iteration break
            if iteration >= self.max_iterations:
                termination_condition = f"Termination: max_iterations ({iteration} == {self.max_iterations})"
                logger.info(termination_condition)

                break

            # consider adding an iterates haven't moved by more than tolerance
            # note that is a finicky termination condition for Benders

        end_time = datetime.datetime.now()

        sp_metadata = self.solutions.metadata
        sp_metadata.iterations = iteration
        sp_metadata.termination_condition = termination_condition
        sp_metadata.start_time = str(start_time)

        variables = [
            solnpool.create_variable(
                value=sp_upper.get_variable_value(b_upper, i),
                index=i,
                name=sp_upper.get_variable_name(i),
            )
            for i, _ in enumerate(sp_upper.get_variables(b=b_upper))
        ]
        obj_value = pyo.value(upper_model.obj)
        objectives = [solnpool.create_objective(value=obj_value)]

        self.solutions.add(variables=variables, objectives=objectives)

        sp_metadata.end_time = str(end_time)
        sp_metadata.time_elapsed = str(end_time - start_time)

        logger.info("")
        logger.info("-" * 70)
        logger.info("BendersSolver - RESULTS")
        for k, v in sp_metadata.items():
            logger.info(f"Metadata {k}: {v}")
        logger.info(f"Objective Value: {objectives[0]}")
        if logger.isEnabledFor(logging.DEBUG):
            pprint.pprint(self.solutions.to_dict())
            sys.stdout.flush()

        logger.info("")
        logger.info("-" * 70)
        logger.info("BendersSolver - STOP")

        return munch.Munch(solutions=self.solutions, upper_model=upper_model)

    def solve(
        self,
        sp_lower,
        eta_bounds_map,
        error_on_initialized_root_vars=False,
        convergence_tol=1e-8,
        subproblem_transforms=None,
        master_transforms=None,
        on_iteration=None,
        **options,
    ):

        data_Munch = self.solve_and_return_model(
            sp_lower=sp_lower,
            eta_bounds_map=eta_bounds_map,
            error_on_initialized_root_vars=error_on_initialized_root_vars,
            convergence_tol=convergence_tol,
            subproblem_transforms=subproblem_transforms,
            master_transforms=master_transforms,
            on_iteration=on_iteration,
            **options,
        )
        return data_Munch.solutions
