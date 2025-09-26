import statistics
import copy
import sys
import munch
import pprint
import numpy as np
import datetime
import logging

try:
    import mpisppy.spin_the_wheel
    import mpisppy.utils.cfg_vanilla
    import mpisppy.agnostic.pyomo_guest
    import mpisppy.agnostic.agnostic_cylinders
    import mpisppy.agnostic.agnostic
    import mpisppy.utils.sputils
    from mpisppy import MPI  # for debugging
    mpisppy_available=True
except:
    mpisppy_available=False

# from pyomo.common.timing import tic, toc, TicTocTimer
from forestlib import solnpool
import forestlib.logs

logger = forestlib.logs.logger


class Forestlib_client:

    def __init__(self, sp):
        self._sp = sp
        self._scenario_probability = {}

    def scenario_creator(self, scenario_name, **kwargs):
        """
        Create the specified scenario
        """

        assert (
            scenario_name in self._scenario_probability
        ), f"Unknown scenario: {scenario_name}"

        # Create the concrete model object
        model = self._sp.create_subproblem(scenario_name)
        model._mpisppy_probability = self._scenario_probability[scenario_name]

        varlist = [
            self._sp.int_to_FirstStageVar[scenario_name][i]
            for i in sorted(self._sp.int_to_FirstStageVar[scenario_name].keys())
        ]
        model._nonant_vardata_list = mpisppy.utils.sputils.build_vardatalist(
            model, varlist
        )
        mpisppy.utils.sputils.attach_root_node(model, 0, varlist)

        return model

    def scenario_names_creator(self, num_scens, start=None):
        bundle_names = sorted(self._sp.bundles.keys())
        assert (
            len(bundle_names) <= num_scens
        ), f"The stochastic program was defined with {len(bundle_names)} bundles, but the user asked for {num_scens} bundles"
        # Keep the the first 'num_scens' bundles
        bundle_names = bundle_names[:num_scens]

        # Compute the per-scenario probability
        total = sum(self._sp.bundles[b].probability for b in bundle_names)
        self._scenario_probability = {
            b: self._sp.bundles[b].probability / total for b in bundle_names
        }

        return bundle_names

    def inparser_adder(self, cfg):
        return cfg

    def kw_creator(self, cfg):
        return {}

    def sample_tree_scen_creator(
        self,
        sname,
        stage,
        sample_branching_factors,
        seed,
        given_scenario=None,
        **scenario_creator_kwargs,
    ):
        """
        Create a scenario within a sample tree. Mainly for multi-stage
        and simple for two-stage. (This function supports zhat and
        confidence interval code)

        Args:
            sname (string):
                scenario name to be created
            stage (int >=1 ):
                for stages > 1, fix data based on sname in earlier stages
            sample_branching_factors (list of ints):
                branching factors for the sample tree
            seed (int):
                To allow random sampling (for some problems, it might be scenario offset)
            given_scenario (Pyomo concrete model):
                if not None, use this to get data for ealier stages
            scenario_creator_kwargs (dict):
                keyword args for the standard scenario creator funcion

        Returns:
            scenario (Pyomo concrete model): A scenario for sname with
            data in stages < stage determined by the arguments
        """
        # Since this is a two-stage problem, we don't have to do much.
        sca = scenario_creator_kwargs.copy()
        sca["seedoffset"] = seed
        sca["num_scens"] = sample_branching_factors[0]  # two-stage problem
        return scenario_creator(sname, **sca)

    def scenario_denouement(self, rank, scenario_name, scenario):
        pass


class Forestlib_guest(mpisppy.agnostic.pyomo_guest.Pyomo_guest):

    def __init__(self, sp):
        self.model_module = Forestlib_client(sp)

    def num_bundles(self):
        return len(self.model_module._sp.bundles)


def mpisppy_agnostic_main(module, Ag, cfg):
    """
    This is the second half of mpisppy.agnostic.agnostic_cylinders.main()
    """
    scenario_creator = Ag.scenario_creator
    assert hasattr(
        module, "scenario_denouement"
    ), "The model file must have a scenario_denouement        function"
    scenario_denouement = module.scenario_denouement  # should we go though Ag?
    # note that if you are bundling, cfg.num_scens will be a fib (numbuns)
    all_scenario_names = module.scenario_names_creator(module.num_bundles())

    # Things needed for vanilla cylinders
    beans = (cfg, scenario_creator, scenario_denouement, all_scenario_names)

    # Vanilla PH hub
    hub_dict = mpisppy.utils.cfg_vanilla.ph_hub(
        *beans,
        scenario_creator_kwargs=None,  # kwargs in Ag not here
        ph_extensions=None,
        ph_converger=None,
        rho_setter=None,
    )
    # pass the Ag object via options...
    hub_dict["opt_kwargs"]["options"]["Ag"] = Ag

    # xhat shuffle bound spoke
    if cfg.xhatshuffle:
        xhatshuffle_spoke = mpisppy.utils.cfg_vanilla.xhatshuffle_spoke(
            *beans, scenario_creator_kwargs=None
        )
        xhatshuffle_spoke["opt_kwargs"]["options"]["Ag"] = Ag
    if cfg.lagrangian:
        lagrangian_spoke = mpisppy.utils.cfg_vanilla.lagrangian_spoke(
            *beans, scenario_creator_kwargs=None
        )
        lagrangian_spoke["opt_kwargs"]["options"]["Ag"] = Ag

    list_of_spoke_dict = list()
    if cfg.xhatshuffle:
        list_of_spoke_dict.append(xhatshuffle_spoke)
    if cfg.lagrangian:
        list_of_spoke_dict.append(lagrangian_spoke)

    wheel = mpisppy.spin_the_wheel.WheelSpinner(hub_dict, list_of_spoke_dict)
    wheel.spin()

    # TODO: Collect the first-stage solution and return it
    # TODO: How do we know if the first stage solution is integral?  How can we force it to be integral?
    # TODO: Collect other statistics from the optimizer (# iterations)

    if cfg.solution_base_name is not None:
        wheel.write_first_stage_solution(f"{cfg.solution_base_name}.csv")
        wheel.write_first_stage_solution(
            f"{cfg.solution_base_name}.npy",
            first_stage_solution_writer=mpisppy.utils.sputils.first_stage_nonant_npy_serializer,
        )
        wheel.write_tree_solution(f"{cfg.solution_base_name}")


def mpisppy_main(sp, options):
    guest = Forestlib_guest(sp)
    cfg = mpisppy.agnostic.agnostic_cylinders._parse_args(guest)
    if options:
        for option,value in options.items():
            cfg[option] = value
    Ag = mpisppy.agnostic.agnostic.Agnostic(guest, cfg)
    mpisppy_agnostic_main(guest, Ag, cfg)


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
                solutions.add(
                    variables=variables,
                    objective=solnpool.Objective(value=sol.objective),
                    suffix=soln.suffix,
                )

    return solutions


class ProgressiveHedgingSolver_MPISPPY(object):

    def __init__(self):
        if mpisppy_available:
            comm = MPI.COMM_WORLD
            self.mpi_rank = comm.Get_rank()
        self.rho = {}
        #self.cached_model_generation = True
        self.max_iterations = 100
        self.time_limit = None
        self.convergence_tolerance = 1e-3
        self.normalize_convergence_norm = True
        self.convergence_norm = 1
        self.solver_name = None
        self.solver_options = {}
        self.finalize_xbar_by_rounding = True
        self.finalize_all_xbar = False
        self.solutions = None
        self.rho_updates = False
        self.default_rho = 1.5

    def set_options(
        self,
        *,
        rho=None,
        #cached_model_generation=None,
        max_iterations=None,
        time_limit=None,
        convergence_tolerance=None,
        normalize_convergence_norm=None,
        convergence_norm=None,
        solver=None,
        solver_options=None,
        loglevel=None,
        finalize_xbar_by_rounding=None,
        finalize_all_xbar=None,
        #solution_manager=None,
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
        #if cached_model_generation is not None:
        #    self.cached_model_generation = cached_model_generation
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if time_limit is not None:
            self.time_limit = time_limit
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
        #if solution_manager is not None:
        #    self.solution_manager = solution_manager

        if loglevel is not None:
            if loglevel == "DEBUG" or loglevel == "VERBOSE":
                forestlib.logs.use_debugging_formatter()
            logger.setLevel(loglevel)

    def solve(self, sp, **options):
        if not mpisppy_available:
            # TODO - return metadata with a useful termination condition
            return None

        start_time = datetime.datetime.now()
        if len(options) > 0:
            self.set_options(**options)


        if self.mpi_rank==0 and logger.isEnabledFor(logging.DEBUG):
            print("Solver Configuration")
            #print(f"  cached_model_generation    {self.cached_model_generation}")
            print(f"  convergence_norm           {self.convergence_norm}")
            print(f"  convergence_tolerance      {self.convergence_tolerance}")
            print(f"  finalize_xbar_by_rounding  {self.finalize_xbar_by_rounding}")
            print(f"  finalize_all_xbar          {self.finalize_all_xbar}")
            print(f"  max_iterations             {self.max_iterations}")
            print(f"  time_limit                 {self.time_limit}")
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
        if self.mpi_rank==0:
            if self.solutions is None:
                self.solutions = solnpool.PoolManager()
            if self.finalize_all_xbar:
                sp_metadata = self.solutions.add_pool("PH Iterations", policy="keep_all")
            else:
                sp_metadata = self.solutions.add_pool("PH Iterations", policy="keep_latest")
            sp_metadata.solver = "PH Iteration Results"
            sp_metadata.solver_options = dict(
                #cached_model_generation=self.cached_model_generation,
                max_iterations=self.max_iterations,
                time_limit=self.time_limit,
                convergence_tolerance=self.convergence_tolerance,
                normalize_convergence_norm=self.normalize_convergence_norm,
                solver_name=self.solver_name,
                solver_options=self.solver_options,
            )

        # The StochProgram object manages the sub-solver interface.  By default, we assume
        #   the user has initialized the sub-solver within the SP object.
        #if self.solver_name:
        #    sp.set_solver(self.solver_name)

        if self.mpi_rank==0:
            logger.info("ProgressiveHedgingSolver_MPISPPY - START")

        options = {
                'solver_name': self.solver_name,
                }
        if self.default_rho:
            options['default_rho'] = self.default_rho
        if self.max_iterations:
            options['max_iterations'] = self.max_iterations
        if self.time_limit:
            options['time_limit'] = self.time_limit
        if self.solver_options:
            options['solver_options'] = self.solver_options
        if logger.isEnabledFor(logging.INFO):
            options['display_progress'] = True
        if logger.isEnabledFor(logging.VERBOSE):
            options['verbose'] = True

        mpisppy_main(sp, options)

        if self.mpi_rank==0:
            end_time = datetime.datetime.now()

            sp_metadata = self.solutions.metadata
            #sp_metadata.iterations = iteration
            #sp_metadata.termination_condition = termination_condition
            sp_metadata.start_time = str(start_time)

            logger.info("")
            logger.info("-" * 70)
            logger.info("ProgressiveHedgingSolver_MPISPPY - FINALIZING")

        if False:
            if self.finalize_all_xbar:
                all_iterations = list(self.solutions)
                self.solutions.add_pool("Finalized All PH Iterations", policy="keep_all")
                for soln in all_iterations:
                    finalize_ph_results(soln, sp=sp, solutions=self.solutions)
            else:
                soln = self.solutions[latest_soln]
                self.solutions.add_pool("Finalized Last PH Solution", policy="keep_best")
                finalize_ph_results(soln, sp=sp, solutions=self.solutions)

        if self.mpi_rank==0:
            sp_metadata.end_time = str(end_time)
            sp_metadata.time_elapsed = str(end_time - start_time)

        if self.mpi_rank == 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("")
                logger.debug("-" * 70)
                logger.debug("ProgressiveHedgingSolver_MPISPPY - RESULTS")
                pprint.pprint(self.solutions.to_dict())
                sys.stdout.flush()

            logger.info("")
            logger.info("-" * 70)
            logger.info("ProgressiveHedgingSolver_MPISPPY - STOP")

            return self.solutions

    def Xlog_iteration(self, **kwds):
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

    def Xarchive_solution(self, *, sp, xbar=None, w=None, **kwds):
        # b = next(iter(sp.bundles))
        variables = [
            solnpool.Variable(
                value=val,
                index=i,
                name=sp.get_variable_name(i),
                suffix=munch.Munch(w={k: v[i] for k, v in w.items()}),
            )
            for i, val in xbar.items()
        ]
        return self.solutions.add(variables=variables, **kwds)
