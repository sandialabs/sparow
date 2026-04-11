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

logger = sparow.logs.logger


class BendersSolver(object):

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
    def _clean_root_model(m,probabilities, eta_bounds = -1_000_000):   
        #TODO: Warning on generic eta bounds

        #TODO: get first_stage_vars from SP information
        #TODO: get first_stage_vars and second_stage_vars in correct format
        #this is a {v.getname(fully_qualified=True): v for v in var_set}
        #TODO: recreate from sp_pyomo, create_ef
        # 4)Constrain First Stage Variable values to be equal under all scenarios
        # access true variable, this is var in var = cuid.find_component_on ...
        # if len(scenarios) > 1:
        #     EF_model.non_ant_cons = pyo.ConstraintList()

        #     for cuid, i in self.varcuid_to_int.items():
        #         for s in scenarios:
        #             var = cuid.find_component_on(EF_model.s[s])
        #             assert (
        #                 var is not None
        #             ), "Pyomo error: Unknown variable '%s' on scenario model '%s'" % (
        #                 cuid,
        #                 s,
        #             )
        #             EF_model.non_ant_cons.add(expr=EF_model.rootx[i] == var)


        first_stage_vars, second_stage_vars = None, None

        #
        # Handle objective logic
        #

        #TODO: Do we have a 'FIRST_STAGE_COST' notion in SP or do I need to build it?
        #if we need to build, ask Bill if there is a better way than the constraint parsing
        m.FIRST_STAGE_COST = 0

        #delete exteraneous objectives:
        for o in m.component_data_objects(pyo.Objective):
            o.deactivate()

        #what is the lower bound in general, how to do this in general
        #so worst case of no action or cost to shed load everything
        #this is model specific, and not having, gets a dual infeasibility
        eta_lower_bound = -1_000_000
        #add eta variable
        #TODO: make k dimensional to match number of scenarios
        m.eta = pyo.Var(bounds=(eta_lower_bound, None))
        
        #TODO: this goes in sp_pyomo for first_stage/second_stage cost expressions
        #assume the objective is linear
        #build from coefficents on linear objective
        #add issue to tolerate someone adding a FIRST_STAGE_COST term
        #add issue to manually decompose non-linear objective
        #email emma to ask if there is a better Pyomo way of doing this

        #set the new objective
        m.master_obj = pyo.Objective(
            #TODO: Update to probability weighted etas
            expr=m.FIRST_STAGE_COST + m.eta,
            sense=pyo.minimize
        )
        #m.master_obj.pprint()


        #
        # Handle constraints
        #

        #need to check if constriant involves terms from second_stage_vars
        #if it does deactivate or delete that constraint
        
        #TODO: create a dict of var : var fully qualified names
        #only call var.getname(fully_qualified=True) if we don't already have it
        # getname fully qualified is expensive and string storage is cheap
        for cons in m.component_data_objects(pyo.Constraint, descend_into=True):
            #variable listing method adapted from
            #https://stackoverflow.com/questions/48538945/access-all-variables-occurring-in-a-pyomo-constraint
            vars = list(pyo.visitor.identify_variables(cons.body))
            remove_cons = False
            for var in vars:
                #TODO: see if we can change this to not in first_stage_vars
                #TODO: check that RHS is dict/set not list
                if var.getname(fully_qualified=True) in second_stage_vars:
                    remove_cons = True
                    break
            if remove_cons:
                cons.deactivate()      

        #
        # Handle second stage variable deletion
        #

        #fix all second stage variables to trivial values
        for var in m.component_data_objects(pyo.Var, descend_into=True):
            if var.getname(fully_qualified=True) in second_stage_vars:
                #TODO: handle edgecases for bound definitions
                #needs to be set to 0, var.lb, var.ub
                #0 if var.lb and var.ub are infinite
                var.fix(0)

        #TODO: make this a Munch agian
        return {'model' : m, 
                'first_stage_names' : list(first_stage_vars.keys()),
                'second_stage_names' : list(second_stage_vars.keys()),
                'first_stage_vars' : list(first_stage_vars.values())}
    
    def _clean_subproblem(*, subproblem, root, first_stage_vars, second_stage_vars):
        if isinstance(first_stage_vars, dict):
            first_stage_vars = list(first_stage_vars.keys())
        if isinstance(second_stage_vars, dict):
            second_stage_vars = list(second_stage_vars.keys())
        #
        # Handle objective logic
        #

        #create general model
        m = subproblem

        #TODO: Do we have a 'SECOND_STAGE_COST' notion in SP or do I need to build it?
        #this is the mirror of the clean master for the first stage cost
        #if we need to build, ask Bill if there is a better way than the constraint parsing
        
        #TODO: handle that this name may not be unique
        #create a unique block for algo stuff and stick the expressions or terms on there
        # or make it m._Benders_Solver_Please_Dont_Conflict_.Other_Thing
        m.SECOND_STAGE_COST = 0

        #delete exteraneous objectives:
        for o in m.component_data_objects(pyo.Objective):
            o.deactivate()

        #set actual objective
        m.subproblem_obj = pyo.Objective(
            expr=m.SECOND_STAGE_COST,
            sense=pyo.minimize
        )
        
        #
        # handle first stage constraints
        #
        #need to check if constriant only invovles terms from first_stage_vars
        #if it does deactivate or delete that constraint

        #TODO: caching of fully qualified names like the clean master subproblem
        for cons in m.component_data_objects(pyo.Constraint, descend_into=True):
            #variable listing method adapted from
            #https://stackoverflow.com/questions/48538945/access-all-variables-occurring-in-a-pyomo-constraint
            vars = list(pyo.visitor.identify_variables(cons.body))
            keep_constraint = False
            for var in vars:
                #TODO: same point here about can we do not in first_stage_vars
                if var.getname(fully_qualified=True) in second_stage_vars:
                    keep_constraint = True
                    break
                #could add a warn line here if var is in neither first or second stage lists
            if not keep_constraint:
                cons.deactivate()


        #
        #   Stagewise linking logic
        #

        #   Build complicating variables map
        #   need to relax domains of first stage vars to reals
        complicating_vars_map = pyo.ComponentMap()
        for name in first_stage_vars:
            root_var = root.find_component(name)
            local_var = m.find_component(name)
            complicating_vars_map[root_var] = local_var
            #relax domain of local first stage var, needed to make dual cut infor work correctly
            #TODO: look at making continuous preserving bounds
            local_var.domain = pyo.Reals
            #TODO: write up a note about EF being over x,y, root over x,theta, subproblem over y, x_tilde with cons add x_tilde=x

        #m.pprint()

        return m, complicating_vars_map

    def solve_in_dev(self, sp, **options):
        #TODO: update iter limit and tolerances
        iter_limit = 1000
        benders_tol = 1e-3


        #
        # Start Generic setup logic
        # same across EF, PH, and Benders
        #
        start_time = datetime.datetime.now()
        if len(options) > 0:
            self.set_options(**options)
        #
        # End Generic setup logic
        #

        #
        # Logging information adapted from PH solver
        # commenting out the finalize and rho lines
        #
        if logger.isEnabledFor(logging.DEBUG):
            print("Solver Configuration")
            print(f"  cached_model_generation    {self.cached_model_generation}")
            print(f"  convergence_norm           {self.convergence_norm}")
            print(f"  convergence_tolerance      {self.convergence_tolerance}")
            #print(f"  finalize_xbar_by_rounding  {self.finalize_xbar_by_rounding}")
            #print(f"  finalize_all_xbar          {self.finalize_all_xbar}")
            print(f"  max_iterations             {self.max_iterations}")
            print(f"  normalize_convergence_norm {self.normalize_convergence_norm}")
            #print(f"  rho                        {self.rho}")
            print(f"  solver_name                {self.solver_name}")
            print("")

        # The StochProgram object manages the sub-solver interface.  By default, we assume
        #   the user has initialized the sub-solver within the SP object.
        if self.solver_name:
            sp.set_solver(self.solver_name)

        logger.info("")
        logger.info("-" * 70)
        logger.info("In Dev BendersSolver - START")
        if logger.isEnabledFor(logging.VERBOSE):
            print(f"  Solver: {self.solver_name}")
            print(f"  Solver Options")
            for k, v in self.solver_options.items():
                print(f"    {k}= {v}")
        tic(None)

        #create subproblems for each of the bundles
        #need it to be 1 scenario to bundle
        #TODO: find out what mode does this
        needed_bundle_mode = "single_scenario"
        sp.initialize_bundles(scheme=needed_bundle_mode)

        num_bundles = len(sp.bundles)
        assert num_bundles > 0, 'Need at least one scenario'

        tic("Initial subproblems", logger=logger, level=logging.VERBOSE)
        subproblems = dict()
        subproblem_probabilities = dict()
        for index, b in enumerate(sp.bundles):
            #TODO: Add issue for sp to create raw subproblem exempt from other caching
            #We are not using that here, but that will explain the concept
            subproblems[index] = sp.create_subproblem(b)
            #TODO: get accurate probabilities
            subproblem_probabilities[index] = sp.bundles[b].probability
            #TODO: add prob info to print statement below
            toc("Created subproblem %s", str(b), logger=logger, level=logging.VERBOSE)
            if index == 0:
                #TODO: this is creating a second copy using the same bundle
                #does this clash with sp having specific bundle based access methods???
                #TODO: check that this M id is different from subproblems[0]
                M = sp.create_subproblem(b)
        #TODO: move the creation of the root model to cloning a created subproblem
        #we can keep the caching on for the rest of the models as subproblems

        #get into contrib.benders format

        #reformat master prob
        #create the theta values for all scenarios, weight by probabilities
        #delete second stage variables and constraints involving second stage vars
        tic("Starting Benders Reformat for master problem", logger=logger, level=logging.VERBOSE)
        #TODO: can we use Bunch/Munch to keep this format
        #yes, you can use Munch here
        #at present this will break because root is a dict not a munch

        #TODO: add issue to tolerate no lower bounds on eta, fix to zero, unfix when there are lower bounds
        #This is future work, we assume lower bounds for now
        root = self._clean_root_model(m = M,probabilities = subproblem_probabilities, eta_bounds = -1_000_000)
        root.model.benders = BendersCutGenerator()
        print(root.first_stage_names)
        print(root.second_stage_names)
        root.model.benders.set_input(root_vars=root.first_stage_vars, tol=1e-3)
        
        toc("Completed Benders Reformat master problem", logger=logger, level=logging.VERBOSE)

        tic("Starting Benders Reformat for subproblems", logger=logger, level=logging.VERBOSE)
        #TODO: handle indexing here
        subproblem_fn_kwargs = dict()
        #TODO: add arg for active subproblem
        subproblem_fn_kwargs['root'] = root.model
        subproblem_fn_kwargs['first_stage_vars'] = root.first_stage_names
        subproblem_fn_kwargs['second_stage_vars'] = root.second_stage_names
        root.model.benders.add_subproblem(
            subproblem_fn=self._clean_subproblem,
            subproblem_fn_kwargs=subproblem_fn_kwargs,
            #TODO: handle indexing of eta
            root_eta=root.model.eta,
            #TODO: address which solver to use, don't assume GLPK
            #Add issue to address this when we are parallelizing cuts
            #Add issue for parallel mode for generating cuts
            subproblem_solver='glpk',
        )
        
        for index in subproblems.keys():
            toc("Created subproblem %s", str(subproblems[index]), logger=logger, level=logging.VERBOSE)

        if logger.isEnabledFor(logging.DEBUG):
            M.pprint()
            M.display()
            sys.stdout.flush()

        toc("Starting Benders Solve", logger=logger, level=logging.VERBOSE)
        i = 0
        #TODO: fix up the interation logic here
        opt = pyo.SolverFactory('glpk')

        for i in range(iter_limit):
            #m.pprint()
            #TODO: confirm this solve is hitting the right model
            res = opt.solve(root.model, tee=False)
            #TODO: check that this works for multiple subproblems
            cuts_added = root.model.benders.generate_cut()
            #TODO: check that cuts added don't need to be added to the cut lists
            #for c in cuts_added:
            #    m.add_constraint(c)
            #name of objective here is user defined
            print(i, len(cuts_added), pyo.value(root.model.master_obj), pyo.value(root.model.eta.value))
            if len(cuts_added) == 0:
                print("Converged")
                break
        #cut loop
            #solve master problem
        tic("Completed Benders Master Problem Solve Iteration %i",i, logger=logger, level=logging.VERBOSE)
            #check convergence
                #add cuts if not converged
        tic("Completed Benders Solve", logger=logger, level=logging.VERBOSE)

        
        # TODO - show value of subproblem
        toc("Optimized extensive form", logger=logger, level=logging.VERBOSE)
        end_time = datetime.datetime.now()

        #TODO: update the benders results info
        benders_termination_condition = None
        benders_status = None
        solutions = or_topas.solnpool.PoolManager()
        metadata = solutions.metadata
        metadata.termination_condition = str(benders_termination_condition)
        metadata.status = str(benders_status)
        metadata.start_time = str(start_time)
        metadata.end_time = str(end_time)
        metadata.time_elapsed = str(end_time - start_time)

        #TODO: handle subproblem variable outputs
        if results.obj_value is not None:
            b = next(iter(sp.bundles))
            #TODO: this interaction with SP seems to imply we can't double up models
            #using the same bundle
            #so the SP structure doesn't appear to be the same block idea as general Pyomo
            #if general blocks, we could just have created one bundle twice
            variables = [
                or_topas.VariableInfo(
                    value=sp.get_variable_value(b, i),
                    index=i,
                    name=sp.get_variable_name(i),
                )
                #TODO: may want to restrict this to first stage variables on master
                for i, _ in enumerate(sp.get_variables())
            ]
            objective = or_topas.ObjectiveInfo(value=results.obj_value)
            solutions.add(variables=variables, objective=objective)

        logger.info("")
        logger.info("-" * 70)
        logger.info("In Dev BendersSolver - STOP")

        return solutions
    
    
