import copy
import sys
import munch
import pprint
import numpy as np

import logging
import forestlib.logs
from forestlib import solnpool

logger = forestlib.logs.logger


def norm(values, p):
    return np.linalg.norm(np.array(values), ord=p)


def finalize_ph_results(soln, *, sp, soln_pool, finalize_xbar_by_rounding=True):
    xbar = [soln.variable(i).value for i in range(len(soln.variables()))]
    assert len(xbar) == len(
        sp.shared_variables()
    ), "Mismatch between solution variables and SP model variables: {len(xbar)} != {len(sp.shared_variables())}"
    #
    # We use xbar to identify a point that is feasible for all scenarios.
    #
    if sp.continuous_fsv():
        logger.info("Finalizing continuous solution")
        #
        # Evaluate the final xbar, and keep if feasible.
        #
        sol = sp.evaluate([xbar[x] for x in sp.shared_variables()])
        if sol.feasible:
            soln_pool.add(
                variables=soln.variables(),
                objective=solnpool.Objective(value=sol.objective),
                suffix=soln.suffix,
            )
    else:
        logger.info("Finalizing solution with binary or integer variables")

        if finalize_xbar_by_rounding:
            #
            # Round the final xbar, and keep if feasible.
            #
            logger.info(
                "\tRounding xbar values associated with binary and integer variables"
            )
            tmpx = [sp.round(x, xbar[x]) for x in sp.shared_variables()]
            sol = sp.evaluate(tmpx)
            if sol.feasible:
                variables = copy.copy(soln.variables())
                for v in variables:
                    v.value = tmpx[v.index]
                soln_pool.add(
                    variables=variables,
                    objective=solnpool.Objective(value=sol.objective),
                    suffix=soln.suffix,
                )

    return soln_pool


class ProgressiveHedgingSolver(object):

    def __init__(self):
        self.rho = 1.5
        self.cached_model_generation = True
        self.max_iterations = 100
        self.convergence_tolerance = 1e-3
        self.normalize_convergence_norm = True
        self.convergence_norm = 1
        self.solver_name = None
        self.solver_options = {}
        self.finalize_xbar_by_rounding = True
        self.finalize_all_xbar = False
        self.solution_pool = None

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
        solution_pool=None,
    ):
        #
        # Misc configuration
        #
        if rho:
            self.rho = rho
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
        if solution_pool is not None:
            self.solution_pool = solution_pool

        if loglevel is not None:
            if loglevel == "DEBUG":
                forestlib.logs.use_debugging_formatter()
            logger.setLevel(loglevel)

    def solve(self, sp, **options):
        if len(options) > 0:
            self.set_options(**options)
        if logger.isEnabledFor(logging.DEBUG):
            print("Solver Configuration")
            print(f"  cached_model_generation    {self.cached_model_generation}")
            print(f"  convergence_norm           {self.convergence_norm}")
            print(f"  convergence_tolerance      {self.convergence_tolerance}")
            print(f"  finalize_xbar_by_rounding  {self.finalize_xbar_by_rounding}")
            print(f"  finalize_all_xbar          {self.finalize_all_xbar}")
            print(f"  max_iterations             {self.max_iterations}")
            print(f"  normalize_convergence_norm {self.normalize_convergence_norm}")
            print(f"  rho                        {self.rho}")
            print(f"  solver_name                {self.solver_name}")
            print("")

        #
        # Setup solution pool and archive context information
        #
        # If finalize_all_xbar is True, then we disable hashing of variables to ensure
        # we keep the solution for each iteration of PH.
        #
        if self.solution_pool is None:
            self.solution_pool = solnpool.SolutionPool()
        sp_metadata = self.solution_pool.set_context(
            "PH Iterations", hash_variables=not self.finalize_all_xbar
        )
        sp_metadata.solver = "PH Iteration Results"
        sp_metadata.solver_options = dict(
            rho=self.rho,
            cached_model_generation=self.cached_model_generation,
            max_iterations=self.max_iterations,
            convergence_tolerance=self.convergence_tolerance,
            normalize_convergence_norm=self.normalize_convergence_norm,
            solver_name=self.solver_name,
            solver_options=self.solver_options,
        )

        # The StochProgram object manages the sub-solver interface.  By default, we assume
        #   the user has initialized the sub-solver within the SP object.
        if self.solver_name:
            sp.set_solver(self.solver_name)

        logger.info("ProgressiveHedgingSolver - START")

        # Step 2
        obj_value = {}
        for b in sp.bundles:
            logger.verbose(f"Creating subproblem '{b}'")
            M = sp.create_subproblem(b, cached=self.cached_model_generation)
            # M.write(f'Iter0_PH_{b}.lp',io_options={'symbolic_solver_labels':True})
            logger.verbose(f"Optimizing subproblem '{b}'")
            results = sp.solve(M, solver_options=self.solver_options)
            assert (
                results.obj_value is not None
            ), f"ERROR solving bundle {b} in initial solve"
            obj_value[b] = results.obj_value
            logger.verbose(f"Optimization Complete")
        obj_lb = sum(sp.bundles[b].probability * obj_value[b] for b in sp.bundles)

        #
        # This is a list of shared first-stage variables amongst all bundles.
        # Note: we need to initialize this here, *after* we create our initial sub-problems.
        #
        sfs_variables = sp.shared_variables()

        # Step 3
        xbar = {}
        for x in sfs_variables:
            xbar[x] = 0.0
            for b in sp.bundles:
                xbar[x] += sp.bundles[b].probability * sp.get_variable_value(b, x)

        # Step 4
        w = {}
        for b in sp.bundles:
            w[b] = {}
            for x in sfs_variables:
                w[b][x] = self.rho * (sp.get_variable_value(b, x) - xbar[x])

        # Step 4.1
        iteration = 0
        termination_condition = "Termination: unknown"
        latest_soln = self.archive_solution(
            sp=sp, xbar=xbar, w=w, iteration=iteration, obj_lb=obj_lb
        )
        self.log_iteration(iteration=iteration, obj_lb=obj_lb, xbar=xbar, rho=self.rho)

        while True:
            iteration += 1

            # Step 5
            xbar_prev = xbar
            w_prev = w

            # Step 6
            obj_value = {}
            for b in sp.bundles:
                logger.verbose(f"Creating subproblem '{b}'")
                logger.debug(f"  b: {b}  w: {w[b]}")
                M = sp.create_subproblem(
                    b=b,
                    w=w_prev[b],
                    x_bar=xbar_prev,
                    rho=self.rho,
                    cached=self.cached_model_generation,
                )
                logger.verbose(f"Optimizing subproblem '{b}'")
                results = sp.solve(M, solver_options=self.solver_options)
                assert (
                    results.obj_value is not None
                ), f"ERROR solving bundle {b} in iteration {iteration}"
                obj_value[b] = results.obj_value
                logger.verbose(f"Optimization Complete")
            obj_lb = sum(sp.bundles[b].probability * obj_value[b] for b in sp.bundles)

            # Step 7
            xbar = {}
            for x in sfs_variables:
                xbar[x] = 0.0
                for b in sp.bundles:
                    logger.debug(
                        f"Variable: {x} {b} {sp.get_variable_name(b,x)} {sp.get_variable_value(b, x)}"
                    )
                    xbar[x] += sp.bundles[b].probability * sp.get_variable_value(b, x)
            logger.debug(f"xbar = {xbar}")

            # Step 8
            w = {}
            for b in sp.bundles:
                w[b] = {}
                for x in sfs_variables:
                    w[b][x] = w_prev[b][x] + self.rho * (
                        sp.get_variable_value(b, x) - xbar[x]
                    )
                logger.debug(f"w[{b}] = {w[b]}")

            # Step 9
            g = 0.0
            for b in sp.bundles:
                g += sp.bundles[b].probability * norm(
                    [sp.get_variable_value(b, x) - xbar[x] for x in sfs_variables],
                    self.convergence_norm,
                )
            if self.normalize_convergence_norm:
                g /= len(sfs_variables)
            logger.info(f"g = {g}")

            # Step 9.1
            tmp = self.archive_solution(
                sp=sp, xbar=xbar, w=w, iteration=iteration, obj_lb=obj_lb, g=g
            )
            if tmp is not None:
                latest_soln = tmp
            self.log_iteration(
                iteration=iteration, obj_lb=obj_lb, xbar=xbar, rho=self.rho
            )

            # Step 10
            if g < self.convergence_tolerance:
                termination_condition = f"Termination: convergence tolerance ({g} < {self.convergence_tolerance})"
                logger.info(termination_condition)
                break

            if iteration >= self.max_iterations:
                termination_condition = f"Termination: max_iterations ({iteration} == {self.max_iterations})"
                logger.info(termination_condition)
                break

            self.update_rho(iteration)

        sp_metadata = self.solution_pool.metadata
        sp_metadata.iterations = iteration
        sp_metadata.termination_condition = termination_condition

        logger.info("")
        logger.info("-" * 70)
        logger.info("ProgressiveHedgingSolver - FINALIZING")
        if self.finalize_all_xbar:
            all_iterations = self.solution_pool.solutions
            # NOTE: we disable hashing here because we want to keep solutions for each iteration
            self.solution_pool.set_context(
                "Finalized All PH Iterations", hash_variables=False
            )
            for soln in all_iterations:
                finalize_ph_results(soln, sp=sp, soln_pool=self.solution_pool)
        else:
            soln = self.solution_pool[latest_soln]
            self.solution_pool.set_context("Finalized Last PH Solution")
            finalize_ph_results(soln, sp=sp, soln_pool=self.solution_pool)

        logger.info("")
        logger.info("-" * 70)
        logger.info("ProgressiveHedgingSolver - RESULTS")
        if logger.level != logging.NOTSET and logger.level <= logging.VERBOSE:
            pprint.pprint(self.solution_pool.to_dict())
            sys.stdout.flush()

        logger.info("")
        logger.info("-" * 70)
        logger.info("ProgressiveHedgingSolver - STOP")

        return self.solution_pool

    def log_iteration(self, **kwds):
        logger.info("")
        logger.info("-" * 70)
        logger.info(f"Iteration:   {kwds['iteration']}")
        logger.info(f"obj_lb:      {kwds['obj_lb']}")
        logger.verbose(f"xbar:        {kwds['xbar']}")
        logger.verbose(f"rho:         {kwds['rho']}")
        logger.info("")

    def archive_solution(self, *, sp, xbar=None, w=None, **kwds):
        b = next(iter(sp.bundles))
        variables = [
            solnpool.Variable(
                value=val,
                index=i,
                name=sp.get_variable_name(b, i),
                suffix=munch.Munch(w={k: v[i] for k, v in w.items()}),
            )
            for i, val in xbar.items()
        ]
        return self.solution_pool.add(variables=variables, **kwds)

    def update_rho(self, iteration):
        # TODO HERE
        pass
