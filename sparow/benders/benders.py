import sys
import numpy as np
import munch
import logging
import datetime

from pyomo.common.timing import tic, toc
import pyomo.environ as pyo
from pyomo.contrib.benders.benders_cuts import BendersCutGenerator
import sparow.logs
import or_topas.solnpool
from pyomo.common.collections import ComponentMap
import pyomo.repn

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
    
    Caveats: 
    1. The probabiilty weights are pushed to the subproblem value functions.
    This allows the master problem to not need to know the probability vector.

    2. Multiple SP objects are created for this process: sp_lower and sp_upper.
    sp_lower has the full scenario information the user inputted.
    sp_upper is created with a single scenario from the information given by the user.
    The bundles in sp_lower will be converted in place into Benders subproblem format.
    The single bundle in sp_upper will be converted in place into the Benders master problem.
    This separates the handling of the |S|+1 models into conceptually separated objects.
    """

    #
    # Init and set_options adapted from ph.py
    #
    def __init__(self):
        self.rho = {}
        self.cached_model_generation = True
        self.max_iterations = 100
        self.convergence_tolerance = 1e-3
        self.normalize_convergence_norm = True
        self.convergence_norm = 1
        self.solver_name = None
        self.solver_options = {}
        self.finalize_xbar_by_rounding = True
        self.finalize_all_xbar = False
        self.solutions = None
        self.rho_updates = False
        self.default_rho = None

    def set_options(
        self,
        *,
        rho=None,
        cached_model_generation=None,
        max_iterations=None,
        convergence_tolerance=None,
        normalize_convergence_norm=None,
        convergence_norm=None,
        solver=None,
        solver_options=None,
        loglevel=None,
        finalize_xbar_by_rounding=None,
        finalize_all_xbar=None,
        solution_manager=None,
        rho_updates=False,
        default_rho=None,
    ):
        # TODO: adapt settings to Benders
        # Likely a subset of these settings for now

        #
        # Misc configuration
        #
        if rho:
            self.rho = rho
        if rho_updates:
            self.rho_updates = rho_updates
        if default_rho:
            self.default_rho = default_rho
        if cached_model_generation is not None:
            self.cached_model_generation = cached_model_generation
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if convergence_tolerance is not None:
            self.convergence_tolerance = convergence_tolerance
        if normalize_convergence_norm is not None:
            self.normalize_convergence_norm = normalize_convergence_norm
        if convergence_norm is not None:
            self.convergence_norm = convergence_norm
        if solver is not None:
            self.solver_name = solver
        if solver_options is not None:
            self.solver_options = solver_options
        if finalize_xbar_by_rounding is not None:
            self.finalize_xbar_by_rounding = finalize_xbar_by_rounding
        if finalize_all_xbar is not None:
            self.finalize_all_xbar = finalize_all_xbar
        if solution_manager is not None:
            self.solution_manager = solution_manager

        if loglevel is not None:
            if loglevel == "DEBUG" or loglevel == "VERBOSE":
                sparow.logs.use_debugging_formatter()
            logger.setLevel(loglevel)

    #
    # This is the same as the ExtensiveForm Solver to test that the logic
    # to access this solver works, separately from the Benders logic
    #
    def solve(self, sp, **options):
        start_time = datetime.datetime.now()
        if len(options) > 0:
            self.set_options(**options)
        # The StochProgram object manages the sub-solver interface.  By default, we assume
        #   the user has initialized the sub-solver within the SP object.
        if self.solver_name:
            sp.set_solver(self.solver_name)

        logger.info("")
        logger.info("-" * 70)
        logger.info("Temp Benders As ExtensiveFormSolver - START")
        if logger.isEnabledFor(logging.VERBOSE):
            print(f"  Solver: {self.solver_name}")
            print(f"  Solver Options")
            for k, v in self.solver_options.items():
                print(f"    {k}= {v}")
        tic(None)

        sp.initialize_bundles(scheme="single_bundle")
        assert (
            len(sp.bundles) == 1
        ), f"The extensive form should only have one bundle: {len(sp.bundles)}"

        b = next(iter(sp.bundles))
        M = sp.create_subproblem(b)
        if logger.isEnabledFor(logging.DEBUG):
            M.pprint()
            M.display()
            sys.stdout.flush()

        toc("Created extensive form", logger=logger, level=logging.VERBOSE)
        results = sp.solve(M, solver_options=self.solver_options)

        # TODO - show value of subproblem
        toc("Optimized extensive form", logger=logger, level=logging.VERBOSE)
        end_time = datetime.datetime.now()

        solutions = or_topas.solnpool.PoolManager()
        metadata = solutions.metadata
        metadata.termination_condition = str(results.termination_condition)
        metadata.status = str(results.status)
        metadata.start_time = str(start_time)
        metadata.end_time = str(end_time)
        metadata.time_elapsed = str(end_time - start_time)

        if results.obj_value is not None:
            b = next(iter(sp.bundles))
            variables = [
                or_topas.VariableInfo(
                    value=sp.get_variable_value(b, i),
                    index=i,
                    name=sp.get_variable_name(i),
                )
                for i, _ in enumerate(sp.get_variables())
            ]
            objective = or_topas.ObjectiveInfo(value=results.obj_value)
            solutions.add(variables=variables, objective=objective)

        logger.info("")
        logger.info("-" * 70)
        logger.info("Temp Benders as ExtensiveFormSolver - STOP")

        return solutions
    
    @staticmethod
    def _transform_to_subproblem_model(sp_lower,b, default_domain= pyo.Reals):
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

        This is done by achieving several steps:
        1. relaxing discrete first-stage variables to continuous,
        2. weighting the objective by probability
        3. remove any exteraneous PH style attributes

        There are optional steps that may be added later:
        4. relaxing second-stage variables as well
        5. ability to set custom domains as a function of which first stage variable it is

        N.B. that this method on its own does not convert into a value function.
        That is done later with OR-TOPAS, where the problem will be modified to make:
        Q_s(bar{x}) := min_{x,y} p_s[<c,x> + <d,y>]
                     s.t.    Ax + By <= h,
                             Ex      <= g,
                             x       == bar{x},
                             x in R^{n}, n = n_1 + n_2
                             y_s in R^{m}        
        """
        assert len(sp_lower.bundles[b].scenarios) == 1, "There should be only one scenario in this bundle"

        #TODO: ask Rachael what the right way to do this construction is
        #might be easier to do with ._create_scenario
        #may need to have W/Rho as None
        #N.B. create_subproblem is a wrapper for create_bundle_ef
        subproblem_model = sp_lower.create_bundle_EF(b, cached=False, w=None, x_bar=None, rho=None, cached=False, compact_repn=True,)
        #we now assume subproblem_model is constructed

        # step 1: probability weight the objective
        #TODO: validate that this objective is not already probability weighted
        #_create_sceanrio does not appear to probability weight
        #all _create_X_bundle_EF_repn does prob weight do on line 455/569/631
        # probability attribute is: self.bundles[b].scenario_probability[s] for s in self.bundles[b].scenarios


        # step 2: relax first_stage variable domains
        for i, x in sp_lower.int_to_FirstStageVar[b].items():
            x.domain = default_domain

        # step 3: remove any unneeded PH style varaible fluff
        # this may be all handled by setting w, x_bar, rho, and cached to false
        return subproblem_model
    
    @staticmethod
    def _create_sp_upper(sp_lower,b):
        """
        This method takes an sp object and a bundle and uses that to make a separate sp object with data
        for just that one bundle b.
        """
        pass

    @staticmethod
    def _transform_to_master_model(*, sp, b, eta_bounds_map, lower_bounding_otherwise_enforced=False, fix_second_stage_vars=False, objective_sense=pyo.minimize, etas_ordered=False,):
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
            A dicitonary mapping scenario keys to bound tuples.
            The tuples are in (LowerBound, UpperBound) format.
            The keys are assumed to be ids for Benders subproblems.
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
        """
        assert objective_sense in (pyo.minimize, pyo.maximize), "Sense needs to be a valid sense"
        assert len(sp.bundles[b].scenarios) == 1, "There should be only one scenario in this bundle"
        assert len(eta_bounds_map) > 0, "Need there to be at least some scenarios to map to"
        for k,v in eta_bounds_map.items():
            assert isinstance(v, tuple) and len(v) == 2, f"Tried to use {v=} as a bound tuple"
            if not lower_bounding_otherwise_enforced:
                assert v[0] is not None, f"Subproblem {k=} must have a lower bound"

        #step 0: create master problem model
        #see discussion in _transform_to_subproblem_model about best method to use here
        upper_model = sp.create_bundle_EF(b, cached=False, w=None, x_bar=None, rho=None, compact_repn=True,)
        #N.B. since this is a single scenario model, it's creation logic will rely on _create_single_scenario_EF_repn
        #the information from that model is then put in EF by ConcreteModel's transfer_attributes_from method
        #so the cuids and maps will be based on this correct model.

        #step 1: compute the objective with just the first stage terms plus constant
        #name the resulting expression the first_stage_cost
        #note the assumption here that the objective is affine
        #this is intentionally done before any of the other alterantions to the model
        repn = pyomo.repn.generate_standard_repn(upper_model.obj, quadratic=False)
        first_stage_cost = repn.constant if repn.constant is not None else 0
        for i, var in enumerate(repn.linear_vars):
            cuid = pyo.ComponentUID(var)
            if cuid in sp.varcuid_to_int:
                #this is the 'is first stage variable' check
                #assumes that all vars in repn.linear_vars come from just the present upper_model
                first_stage_cost += repn.linear_coefs[i]*var

        #step 2: delete all constraints involving second-stage variables and second-stage variables
        #this removes all the recourse/scenario specific feasibility constraints/variables

        #gather cons to remove to process later
        #TODO: consider caching the CUIDs for second-stage vars
        cons_to_delete = []
        for cons in upper_model.component_data_objects(pyo.Constraint, descend_into=True):
            #variable listing method adapted from
            #https://stackoverflow.com/questions/48538945/access-all-variables-occurring-in-a-pyomo-constraint
            if any(pyo.ComponentUID(var, context=upper_model) not in sp.varcuid_to_int for var in pyo.visitor.identify_variables(cons.body)):
                #this amounts to an any vars not in first_stage_vars check
                # sp.varcuid_to_int only holds varcuid's for first_stage_variables
                cons_to_delete.append(cons)

        #unneed cons handling point
        for cons in cons_to_delete:
            #we can either deactivate or delete
            #deactivating for now
            cons.deactivate()

        #unneeded vars handling point
        if fix_second_stage_vars:
            #gather vars to remove to process later
            vars_to_delete = []
            for var in upper_model.component_data_objects(pyo.Var, descend_into=True):
                # this amounts to a var not in first_stage_vars check
                # sp.varcuid_to_int only holds varcuid's for first_stage_variables
                if pyo.ComponentUID(var, context=upper_model) not in sp.varcuid_to_int:
                    #TODO: handle edgecases for bound definitions
                    vars_to_delete.append(var)

            for var in vars_to_delete:
                #we can either fix or delete
                #fixing for now because we do not want unneed slack variables when generating alternative solutions
                var.domain = pyo.Reals
                var.fix(0)

        #step 3: create eta tracking variables for the scenario set (either 1...N or a pyomo set)
        #enforce that there are upper and lower bound entries in the eta_bounds dict for each scenario
        #none corresponds to no bound on that side
        #feasibility only problems can be set to have eta_i = 0 by LB_i = UB_i = 0
        upper_model.scenarios = pyo.Set(initialize=list(eta_bounds_map.keys()), ordered=etas_ordered)
        def eta_bound_rule(m, s):
            return eta_bounds_map[s]
        upper_model.etas = pyo.Var(upper_model.scenarios, rule = eta_bound_rule)

        #step 4: add eta.sum to the objective expression
        if hasattr(upper_model, 'obj') and hasattr(upper_model.obj, 'active') and upper_model.obj.active:
            upper_model.obj.deactivate()
        upper_model.obj = pyo.Objective(expr=first_stage_cost + upper_model.etas.sum(), sense=objective_sense)
        
        return upper_model

    @staticmethod
    def _setup_topas_subproblem(sp_lower, b_lower, sp_upper, b_upper):
        """
        This is a wrapper method that takes a stochastic program and bundle, sp_lower and b_lower,
        and returns data for use as an OR-TOPAS Benders subproblem.

        At present, it builds the needed map from the first-stage variables in m_upper to those in b.
        It then returns the subproblem model, b, and that complicating variable map.
        """

        #rely on _transform_to_subproblem_model to create the subproblem model
        model_lower = BendersSolver._transform_to_subproblem_model(sp_lower,b_lower, default_domain= pyo.Reals)

        #create the complicating variable map
        #use the sp objects both having first stage variable lists to build the mapping

        #so self.int_to_FirstStageVarName is not indexed by bundles, so for compact_repn models
        #it should be identical between different bundles (and objects created from the same data)
        #it also means that the component map should be
        #this format actually does not require that sp_upper and sp_lower are separate sp objects, but it can handle separate sp objects

        complicating_variable_map = ComponentMap()
        for i, var_upper in sp_upper.int_to_FirstStageVar[b_upper].items():
            complicating_variable_map[var_upper] = sp_lower.int_to_FirstStageVar[b_lower][i]

        #return the subproblem model, b, and the complicating variable map.
        return model_lower, complicating_variable_map

    #steps for general solve
    #take overall sp model with full scenario data, this will be sp_lower
    #create sp model with single scenario for sp_upper
    #for each bundle in sp_lower, transform to subproblem
    #transform sp_upper model to master format
    #setup or-topas benders model
        # extract subproblem solver information from sp_lower
    #iterate while adding benders cuts
    #report results to user

    # def _clean_root_model(m,probabilities, eta_bounds = -1_000_000):   
    #     #TODO: Warning on generic eta bounds

    #     #TODO: get first_stage_vars from SP information
    #     #TODO: get first_stage_vars and second_stage_vars in correct format
    #     #this is a {v.getname(fully_qualified=True): v for v in var_set}
    #     #TODO: recreate from sp_pyomo, create_ef
    #     # 4)Constrain First Stage Variable values to be equal under all scenarios
    #     # access true variable, this is var in var = cuid.find_component_on ...
    #     # if len(scenarios) > 1:
    #     #     EF_model.non_ant_cons = pyo.ConstraintList()

    #     #     for cuid, i in self.varcuid_to_int.items():
    #     #         for s in scenarios:
    #     #             var = cuid.find_component_on(EF_model.s[s])
    #     #             assert (
    #     #                 var is not None
    #     #             ), "Pyomo error: Unknown variable '%s' on scenario model '%s'" % (
    #     #                 cuid,
    #     #                 s,
    #     #             )
    #     #             EF_model.non_ant_cons.add(expr=EF_model.rootx[i] == var)


    #     first_stage_vars, second_stage_vars = None, None

    #     #
    #     # Handle objective logic
    #     #

    #     #TODO: Do we have a 'FIRST_STAGE_COST' notion in SP or do I need to build it?
    #     #if we need to build, ask Bill if there is a better way than the constraint parsing
    #     m.FIRST_STAGE_COST = 0

    #     #delete exteraneous objectives:
    #     for o in m.component_data_objects(pyo.Objective):
    #         o.deactivate()

    #     #what is the lower bound in general, how to do this in general
    #     #so worst case of no action or cost to shed load everything
    #     #this is model specific, and not having, gets a dual infeasibility
    #     eta_lower_bound = -1_000_000
    #     #add eta variable
    #     #TODO: make k dimensional to match number of scenarios
    #     m.eta = pyo.Var(bounds=(eta_lower_bound, None))
        
    #     #TODO: this goes in sp_pyomo for first_stage/second_stage cost expressions
    #     #assume the objective is linear
    #     #build from coefficents on linear objective
    #     #add issue to tolerate someone adding a FIRST_STAGE_COST term
    #     #add issue to manually decompose non-linear objective
    #     #email emma to ask if there is a better Pyomo way of doing this

    #     #set the new objective
    #     m.master_obj = pyo.Objective(
    #         #TODO: Update to probability weighted etas
    #         expr=m.FIRST_STAGE_COST + m.eta,
    #         sense=pyo.minimize
    #     )
    #     #m.master_obj.pprint()


    #     #
    #     # Handle constraints
    #     #

    #     #need to check if constriant involves terms from second_stage_vars
    #     #if it does deactivate or delete that constraint
        
    #     #TODO: create a dict of var : var fully qualified names
    #     #only call var.getname(fully_qualified=True) if we don't already have it
    #     # getname fully qualified is expensive and string storage is cheap
    #     for cons in m.component_data_objects(pyo.Constraint, descend_into=True):
    #         #variable listing method adapted from
    #         #https://stackoverflow.com/questions/48538945/access-all-variables-occurring-in-a-pyomo-constraint
    #         vars = list(pyo.visitor.identify_variables(cons.body))
    #         remove_cons = False
    #         for var in vars:
    #             #TODO: see if we can change this to not in first_stage_vars
    #             #TODO: check that RHS is dict/set not list
    #             if var.getname(fully_qualified=True) in second_stage_vars:
    #                 remove_cons = True
    #                 break
    #         if remove_cons:
    #             cons.deactivate()      

    #     #
    #     # Handle second stage variable deletion
    #     #

    #     #fix all second stage variables to trivial values
    #     for var in m.component_data_objects(pyo.Var, descend_into=True):
    #         if var.getname(fully_qualified=True) in second_stage_vars:
    #             #TODO: handle edgecases for bound definitions
    #             #needs to be set to 0, var.lb, var.ub
    #             #0 if var.lb and var.ub are infinite
    #             var.fix(0)

    #     #TODO: make this a Munch agian
    #     return {'model' : m, 
    #             'first_stage_names' : list(first_stage_vars.keys()),
    #             'second_stage_names' : list(second_stage_vars.keys()),
    #             'first_stage_vars' : list(first_stage_vars.values())}
    
    # def _clean_subproblem(*, subproblem, root, first_stage_vars, second_stage_vars):
    #     if isinstance(first_stage_vars, dict):
    #         first_stage_vars = list(first_stage_vars.keys())
    #     if isinstance(second_stage_vars, dict):
    #         second_stage_vars = list(second_stage_vars.keys())
    #     #
    #     # Handle objective logic
    #     #

    #     #create general model
    #     m = subproblem

    #     #TODO: Do we have a 'SECOND_STAGE_COST' notion in SP or do I need to build it?
    #     #this is the mirror of the clean master for the first stage cost
    #     #if we need to build, ask Bill if there is a better way than the constraint parsing
        
    #     #TODO: handle that this name may not be unique
    #     #create a unique block for algo stuff and stick the expressions or terms on there
    #     # or make it m._Benders_Solver_Please_Dont_Conflict_.Other_Thing
    #     m.SECOND_STAGE_COST = 0

    #     #delete exteraneous objectives:
    #     for o in m.component_data_objects(pyo.Objective):
    #         o.deactivate()

    #     #set actual objective
    #     m.subproblem_obj = pyo.Objective(
    #         expr=m.SECOND_STAGE_COST,
    #         sense=pyo.minimize
    #     )
        
    #     #
    #     # handle first stage constraints
    #     #
    #     #need to check if constriant only invovles terms from first_stage_vars
    #     #if it does deactivate or delete that constraint

    #     #TODO: caching of fully qualified names like the clean master subproblem
    #     for cons in m.component_data_objects(pyo.Constraint, descend_into=True):
    #         #variable listing method adapted from
    #         #https://stackoverflow.com/questions/48538945/access-all-variables-occurring-in-a-pyomo-constraint
    #         vars = list(pyo.visitor.identify_variables(cons.body))
    #         keep_constraint = False
    #         for var in vars:
    #             #TODO: same point here about can we do not in first_stage_vars
    #             if var.getname(fully_qualified=True) in second_stage_vars:
    #                 keep_constraint = True
    #                 break
    #             #could add a warn line here if var is in neither first or second stage lists
    #         if not keep_constraint:
    #             cons.deactivate()


    #     #
    #     #   Stagewise linking logic
    #     #

    #     #   Build complicating variables map
    #     #   need to relax domains of first stage vars to reals
    #     complicating_vars_map = pyo.ComponentMap()
    #     for name in first_stage_vars:
    #         root_var = root.find_component(name)
    #         local_var = m.find_component(name)
    #         complicating_vars_map[root_var] = local_var
    #         #relax domain of local first stage var, needed to make dual cut infor work correctly
    #         #TODO: look at making continuous preserving bounds
    #         local_var.domain = pyo.Reals
    #         #TODO: write up a note about EF being over x,y, root over x,theta, subproblem over y, x_tilde with cons add x_tilde=x

    #     #m.pprint()

    #     return m, complicating_vars_map

    # def solve_in_dev(self, sp, **options):
    #     #TODO: update iter limit and tolerances
    #     iter_limit = 1000
    #     benders_tol = 1e-3


    #     #
    #     # Start Generic setup logic
    #     # same across EF, PH, and Benders
    #     #
    #     start_time = datetime.datetime.now()
    #     if len(options) > 0:
    #         self.set_options(**options)
    #     #
    #     # End Generic setup logic
    #     #

    #     #
    #     # Logging information adapted from PH solver
    #     # commenting out the finalize and rho lines
    #     #
    #     if logger.isEnabledFor(logging.DEBUG):
    #         print("Solver Configuration")
    #         print(f"  cached_model_generation    {self.cached_model_generation}")
    #         print(f"  convergence_norm           {self.convergence_norm}")
    #         print(f"  convergence_tolerance      {self.convergence_tolerance}")
    #         #print(f"  finalize_xbar_by_rounding  {self.finalize_xbar_by_rounding}")
    #         #print(f"  finalize_all_xbar          {self.finalize_all_xbar}")
    #         print(f"  max_iterations             {self.max_iterations}")
    #         print(f"  normalize_convergence_norm {self.normalize_convergence_norm}")
    #         #print(f"  rho                        {self.rho}")
    #         print(f"  solver_name                {self.solver_name}")
    #         print("")

    #     # The StochProgram object manages the sub-solver interface.  By default, we assume
    #     #   the user has initialized the sub-solver within the SP object.
    #     if self.solver_name:
    #         sp.set_solver(self.solver_name)

    #     logger.info("")
    #     logger.info("-" * 70)
    #     logger.info("In Dev BendersSolver - START")
    #     if logger.isEnabledFor(logging.VERBOSE):
    #         print(f"  Solver: {self.solver_name}")
    #         print(f"  Solver Options")
    #         for k, v in self.solver_options.items():
    #             print(f"    {k}= {v}")
    #     tic(None)

    #     #create subproblems for each of the bundles
    #     #need it to be 1 scenario to bundle
    #     #TODO: find out what mode does this
    #     needed_bundle_mode = "single_scenario"
    #     sp.initialize_bundles(scheme=needed_bundle_mode)

    #     num_bundles = len(sp.bundles)
    #     assert num_bundles > 0, 'Need at least one scenario'

    #     tic("Initial subproblems", logger=logger, level=logging.VERBOSE)
    #     subproblems = dict()
    #     subproblem_probabilities = dict()
    #     for index, b in enumerate(sp.bundles):
    #         #TODO: Add issue for sp to create raw subproblem exempt from other caching
    #         #We are not using that here, but that will explain the concept
    #         subproblems[index] = sp.create_subproblem(b)
    #         #TODO: get accurate probabilities
    #         subproblem_probabilities[index] = sp.bundles[b].probability
    #         #TODO: add prob info to print statement below
    #         toc("Created subproblem %s", str(b), logger=logger, level=logging.VERBOSE)
    #         if index == 0:
    #             #TODO: this is creating a second copy using the same bundle
    #             #does this clash with sp having specific bundle based access methods???
    #             #TODO: check that this M id is different from subproblems[0]
    #             M = sp.create_subproblem(b)
    #     #TODO: move the creation of the root model to cloning a created subproblem
    #     #we can keep the caching on for the rest of the models as subproblems

    #     #get into contrib.benders format

    #     #reformat master prob
    #     #create the theta values for all scenarios, weight by probabilities
    #     #delete second stage variables and constraints involving second stage vars
    #     tic("Starting Benders Reformat for master problem", logger=logger, level=logging.VERBOSE)
    #     #TODO: can we use Bunch/Munch to keep this format
    #     #yes, you can use Munch here
    #     #at present this will break because root is a dict not a munch

    #     #TODO: add issue to tolerate no lower bounds on eta, fix to zero, unfix when there are lower bounds
    #     #This is future work, we assume lower bounds for now
    #     root = self._clean_root_model(m = M,probabilities = subproblem_probabilities, eta_bounds = -1_000_000)
    #     root.model.benders = BendersCutGenerator()
    #     print(root.first_stage_names)
    #     print(root.second_stage_names)
    #     root.model.benders.set_input(root_vars=root.first_stage_vars, tol=1e-3)
        
    #     toc("Completed Benders Reformat master problem", logger=logger, level=logging.VERBOSE)

    #     tic("Starting Benders Reformat for subproblems", logger=logger, level=logging.VERBOSE)
    #     #TODO: handle indexing here
    #     subproblem_fn_kwargs = dict()
    #     #TODO: add arg for active subproblem
    #     subproblem_fn_kwargs['root'] = root.model
    #     subproblem_fn_kwargs['first_stage_vars'] = root.first_stage_names
    #     subproblem_fn_kwargs['second_stage_vars'] = root.second_stage_names
    #     root.model.benders.add_subproblem(
    #         subproblem_fn=self._clean_subproblem,
    #         subproblem_fn_kwargs=subproblem_fn_kwargs,
    #         #TODO: handle indexing of eta
    #         root_eta=root.model.eta,
    #         #TODO: address which solver to use, don't assume GLPK
    #         #Add issue to address this when we are parallelizing cuts
    #         #Add issue for parallel mode for generating cuts
    #         subproblem_solver='glpk',
    #     )
        
    #     for index in subproblems.keys():
    #         toc("Created subproblem %s", str(subproblems[index]), logger=logger, level=logging.VERBOSE)

    #     if logger.isEnabledFor(logging.DEBUG):
    #         M.pprint()
    #         M.display()
    #         sys.stdout.flush()

    #     toc("Starting Benders Solve", logger=logger, level=logging.VERBOSE)
    #     i = 0
    #     #TODO: fix up the interation logic here
    #     opt = pyo.SolverFactory('glpk')

    #     for i in range(iter_limit):
    #         #m.pprint()
    #         #TODO: confirm this solve is hitting the right model
    #         res = opt.solve(root.model, tee=False)
    #         #TODO: check that this works for multiple subproblems
    #         cuts_added = root.model.benders.generate_cut()
    #         #TODO: check that cuts added don't need to be added to the cut lists
    #         #for c in cuts_added:
    #         #    m.add_constraint(c)
    #         #name of objective here is user defined
    #         print(i, len(cuts_added), pyo.value(root.model.master_obj), pyo.value(root.model.eta.value))
    #         if len(cuts_added) == 0:
    #             print("Converged")
    #             break
    #     #cut loop
    #         #solve master problem
    #     tic("Completed Benders Master Problem Solve Iteration %i",i, logger=logger, level=logging.VERBOSE)
    #         #check convergence
    #             #add cuts if not converged
    #     tic("Completed Benders Solve", logger=logger, level=logging.VERBOSE)

        
    #     # TODO - show value of subproblem
    #     toc("Optimized extensive form", logger=logger, level=logging.VERBOSE)
    #     end_time = datetime.datetime.now()

    #     #TODO: update the benders results info
    #     benders_termination_condition = None
    #     benders_status = None
    #     solutions = or_topas.solnpool.PoolManager()
    #     metadata = solutions.metadata
    #     metadata.termination_condition = str(benders_termination_condition)
    #     metadata.status = str(benders_status)
    #     metadata.start_time = str(start_time)
    #     metadata.end_time = str(end_time)
    #     metadata.time_elapsed = str(end_time - start_time)

    #     #TODO: handle subproblem variable outputs
    #     if results.obj_value is not None:
    #         b = next(iter(sp.bundles))
    #         #TODO: this interaction with SP seems to imply we can't double up models
    #         #using the same bundle
    #         #so the SP structure doesn't appear to be the same block idea as general Pyomo
    #         #if general blocks, we could just have created one bundle twice
    #         variables = [
    #             or_topas.VariableInfo(
    #                 value=sp.get_variable_value(b, i),
    #                 index=i,
    #                 name=sp.get_variable_name(i),
    #             )
    #             #TODO: may want to restrict this to first stage variables on master
    #             for i, _ in enumerate(sp.get_variables())
    #         ]
    #         objective = or_topas.ObjectiveInfo(value=results.obj_value)
    #         solutions.add(variables=variables, objective=objective)

    #     logger.info("")
    #     logger.info("-" * 70)
    #     logger.info("In Dev BendersSolver - STOP")

    #     return solutions
    
    
