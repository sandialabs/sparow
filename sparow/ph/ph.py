import statistics
import copy
import sys
import munch
import pprint
import numpy as np
import datetime
import logging

from pyomo.common.timing import tic, toc, TicTocTimer
import sparow.logs
import or_topas

logger = sparow.logs.logger


def norm(values, p):
    return np.linalg.norm(np.array(values), ord=p)


def finalize_ph_results(soln, *, sp, solutions, finalize_xbar_by_rounding=True):
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
            solutions.add(
                variables=soln.variables(),
                objective=or_topas.ObjectiveInfo(value=sol.objective),
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
                solutions.add(
                    variables=variables,
                    objective=or_topas.ObjectiveInfo(value=sol.objective),
                    suffix=soln.suffix,
                )

    return solutions


class ProgressiveHedgingSolver(object):

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

    def solve(self, sp, **options):
        start_time = datetime.datetime.now()
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
        # Setup solution manager and archive context information
        #
        # If finalize_all_xbar is True, then we disable hashing of variables to ensure
        # we keep the solution for each iteration of PH.
        #
        if self.solutions is None:
            self.solutions = or_topas.PoolManager()
        if self.finalize_all_xbar:
            sp_metadata = self.solutions.add_pool(name="PH Iterations", policy=or_topas.PoolPolicy.keep_all)
        else:
            sp_metadata = self.solutions.add_pool(name="PH Iterations", policy=or_topas.PoolPolicy.keep_latest)
        sp_metadata.solver = "PH Iteration Results"
        sp_metadata.solver_options = dict(
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
        iteration_timer = TicTocTimer()
        iteration_timer.tic(None)

        # Step 2
        obj_value = {}
        tic("Initial subproblems", logger=logger, level=logging.VERBOSE)
        for b in sp.bundles:
            M = sp.create_subproblem(b, cached=self.cached_model_generation)
            toc("Created subproblem %s", str(b), logger=logger, level=logging.VERBOSE)

            results = sp.solve(M, solver_options=self.solver_options)
            assert (
                results.obj_value is not None
            ), f"ERROR solving bundle {b} in initial solve"
            obj_value[b] = results.obj_value
            toc("Optimized subproblem %s", str(b), logger=logger, level=logging.VERBOSE)
        obj_lb = sum(sp.bundles[b].probability * obj_value[b] for b in sp.bundles)

        #
        # This is a list of shared first-stage variables amongst all bundles.
        # Note: we need to initialize this here, *after* we create our initial sub-problems.
        #
        sfs_variables = sp.shared_variables()

        # Step 3
        xbar = {}
        for x in sfs_variables:
            xbar[x] = sum(
                sp.bundles[b].probability * sp.get_variable_value(b, x)
                for b in sp.bundles
            )
        self.update_rho(sfs_variables, xbar, sp)

        # Step 4
        w = {}
        for b in sp.bundles:
            w[b] = {}
            for x in sfs_variables:
                w[b][x] = self.rho[x] * (sp.get_variable_value(b, x) - xbar[x])

        # Step 4.1
        iteration = 0
        termination_condition = "Termination: unknown"
        latest_soln = self.archive_solution(
            sp=sp, xbar=xbar, w=w, iteration=iteration, obj_lb=obj_lb
        )
        time_last_iter = iteration_timer.toc(None)
        self.log_iteration(
            iteration=iteration,
            obj_lb=obj_lb,
            time=datetime.datetime.now(),
            time_last_iter=time_last_iter,
            xbar=xbar,
            rho=self.rho,
            w=w,
        )

        while True:
            iteration_timer.tic(None)
            iteration += 1

            # Step 5
            xbar_prev = xbar
            w_prev = w

            # Step 6
            tic(
                f"Subproblems for iteration {iteration}",
                logger=logger,
                level=logging.VERBOSE,
            )
            obj_value = {}
            for b in sp.bundles:
                logger.debug(f"  b: {b}  w: {w[b]}")
                M = sp.create_subproblem(
                    b=b,
                    w=w_prev[b],
                    x_bar=xbar_prev,
                    rho=self.rho,
                    cached=self.cached_model_generation,
                )
                toc(
                    "Created subproblem %s",
                    str(b),
                    logger=logger,
                    level=logging.VERBOSE,
                )

                results = sp.solve(M, solver_options=self.solver_options)
                assert (
                    results.obj_value is not None
                ), f"ERROR solving bundle {b} in iteration {iteration}"
                obj_value[b] = results.obj_value
                toc(
                    "Optimized subproblem %s",
                    str(b),
                    logger=logger,
                    level=logging.VERBOSE,
                )
            obj_lb = sum(sp.bundles[b].probability * obj_value[b] for b in sp.bundles)

            # Step 7
            xbar = {}
            for x in sfs_variables:
                xbar[x] = sum(
                    sp.bundles[b].probability * sp.get_variable_value(b, x)
                    for b in sp.bundles
                )
                for b in sp.bundles:
                    logger.debug(
                        f"Variable: {x} {b} {sp.get_variable_name(x)} {sp.get_variable_value(b, x)}"
                    )
            logger.debug(f"xbar = {xbar}")

            self.update_rho(sfs_variables, xbar, sp)
            logger.debug(f"rho = {self.rho}")

            # Step 8
            w = {}
            for b in sp.bundles:
                w[b] = {}
                for x in sfs_variables:
                    w[b][x] = w_prev[b][x] + self.rho[x] * (
                        sp.get_variable_value(b, x) - xbar[x]
                    )
                logger.debug(f"w[{b}] = {w[b]}")

            # Step 9
            #
            # NOTE: Should we use xbar_prev instead of xbar to compute 'g'?  If you use xbar, then
            #       all subproblems could have the same value, but in the *next* iteration they
            #       subproblems could have different values.  We need to assess the difference
            #       between the xbar that generated the current values.
            #
            g = sum(
                sp.bundles[b].probability
                * norm(
                    [sp.get_variable_value(b, x) - xbar[x] for x in sfs_variables],
                    self.convergence_norm,
                )
                for b in sp.bundles
            )
            if self.normalize_convergence_norm:
                g /= len(sfs_variables)
            logger.info(f"g = {g}")

            G = norm(
                [xbar[x] - xbar_prev[x] for x in sfs_variables], self.convergence_norm
            )
            if self.normalize_convergence_norm:
                G /= len(sfs_variables)
            logger.info(f"G = {G}")

            # Step 9.1
            tic(
                f"Archiving solution: {iteration}", logger=logger, level=logging.VERBOSE
            )
            latest_soln = self.archive_solution(
                sp=sp, xbar=xbar, w=w, iteration=iteration, obj_lb=obj_lb, g=g
            )
            toc(f"Archiving solution - DONE", logger=logger, level=logging.VERBOSE)

            time_last_iter = iteration_timer.toc(None)
            self.log_iteration(
                iteration=iteration,
                obj_lb=obj_lb,
                time=datetime.datetime.now(),
                time_last_iter=time_last_iter,
                xbar=xbar,
                rho=self.rho,
                g=g,
                G=G,
                w=w,
            )

            # Step 10
            if G + g < self.convergence_tolerance:
                termination_condition = f"Termination: convergence tolerance ({G} + {g} < {self.convergence_tolerance})"
                logger.info(termination_condition)
                break

            if iteration >= self.max_iterations:
                termination_condition = f"Termination: max_iterations ({iteration} == {self.max_iterations})"
                logger.info(termination_condition)
                break

        end_time = datetime.datetime.now()

        sp_metadata = self.solutions.metadata
        sp_metadata.iterations = iteration
        sp_metadata.termination_condition = termination_condition
        sp_metadata.start_time = str(start_time)

        logger.info("")
        logger.info("-" * 70)
        logger.info("ProgressiveHedgingSolver - FINALIZING")
        if self.finalize_all_xbar:
            all_iterations = list(self.solutions)
            self.solutions.add_pool(name="Finalized All PH Iterations", policy=or_topas.PoolPolicy.keep_all)
            for soln in all_iterations:
                finalize_ph_results(soln, sp=sp, solutions=self.solutions)
        else:
            soln = self.solutions[latest_soln]
            self.solutions.add_pool(name="Finalized Last PH Solution", policy=or_topas.PoolPolicy.keep_best)
            finalize_ph_results(soln, sp=sp, solutions=self.solutions)

        sp_metadata.end_time = str(end_time)
        sp_metadata.time_elapsed = str(end_time - start_time)

        logger.info("")
        logger.info("-" * 70)
        logger.info("ProgressiveHedgingSolver - RESULTS")
        if logger.isEnabledFor(logging.DEBUG):
            pprint.pprint(self.solutions.to_dict())
            sys.stdout.flush()

        logger.info("")
        logger.info("-" * 70)
        logger.info("ProgressiveHedgingSolver - STOP")

        return self.solutions

    def log_iteration(self, **kwds):
        logger.info("")
        logger.info("-" * 70)
        logger.info(f"Iteration:        {kwds['iteration']}")
        logger.info(f"obj_lb:           {kwds['obj_lb']}")
        logger.info(f"conv_norm:        {kwds.get('g',None)}")
        logger.info(f"xbar_diff_norm:   {kwds.get('G',None)}")
        logger.info(f"time:             {kwds['time']}")
        logger.info(f"time_last_iter:   {kwds['time_last_iter']}")
        if logger.isEnabledFor(logging.VERBOSE):
            tmp = kwds["w"]
            tmp = {
                k: statistics.mean(abs(val) for val in v.values())
                for k, v in tmp.items()
            }
            if len(tmp) > 10:
                _vals = list(tmp.values())
                logger.verbose(f"w_min:            {min(_vals)}")
                logger.verbose(f"w_mean:           {statistics.mean(_vals)}")
                logger.verbose(f"w_max:            {max(_vals)}")
            else:
                logger.verbose(f"w_mean_abs:       {tmp}")

            tmp = kwds["xbar"]
            if len(tmp) > 10:
                _vals = list(abs(v) for v in tmp.values())
                logger.verbose(f"xbar_min_abs:     {min(_vals)}")
                logger.verbose(f"xbar_mean_abs:    {statistics.mean(_vals)}")
                logger.verbose(f"xbar_max_abs:     {max(_vals)}")
            else:
                tmp = {k: v for k, v in tmp.items() if v != 0}
                logger.verbose(f"xbar_abs:         {tmp}")

            tmp = kwds["rho"]
            if len(tmp) > 10:
                _vals = list(tmp.values())
                logger.verbose(f"rho_min:          {min(_vals)}")
                logger.verbose(f"rho_mean:         {statistics.mean(_vals)}")
                logger.verbose(f"rho_max:          {max(_vals)}")
            else:
                logger.verbose(f"rho:              {tmp}")
        logger.info("")

    def archive_solution(self, *, sp, xbar=None, w=None, **kwds):
        # b = next(iter(sp.bundles))
        variables = [
            or_topas.VariableInfo(
                value=val,
                index=i,
                name=sp.get_variable_name(i),
                suffix=munch.Munch(w={k: v[i] for k, v in w.items()}),
            )
            for i, val in xbar.items()
        ]
        return self.solutions.add(variables=variables, **kwds)

    def update_rho(self, sfs_variables, xbar, sp):
        # this function is scenario-independent, but will need to be updated for integer x
        if self.rho_updates:
            for x in sfs_variables:
                if abs(sp.get_objective_coef(x)) > 0:
                    self.rho[x] = abs(sp.get_objective_coef(x)) / max(
                        sum(
                            sp.bundles[b].probability
                            * abs(sp.get_variable_value(b, x) - xbar[x])
                            for b in sp.bundles
                        ),
                        1,
                    )
                else:
                    if self.default_rho:
                        self.rho[x] = self.default_rho
                    else:
                        self.rho[x] = 1.5
                        logger.warning(
                            f"Variable objective coefficient is 0; rho{x} set to 1.5"
                        )
        else:
            for x in sfs_variables:
                if self.default_rho:
                    self.rho[x] = self.default_rho
                else:
                    self.rho[x] = 1.5
