import statistics
import copy
import sys
import munch
import pprint
import numpy as np
import datetime
import logging
from functools import partial

try:
    import mpisppy.utils.sputils
    from mpisppy import MPI  # for debugging

    mpisppy_available = True
except:
    mpisppy_available = False

import pyomo.environ as pyo

# from pyomo.common.timing import tic, toc, TicTocTimer
from forestlib import solnpool
from forestlib.sp.sp_pyomo import find_objective
import forestlib.logs

logger = forestlib.logs.logger


class Forestlib_client:

    def __init__(self, sp):
        self._sp = sp
        self._scenario_probability = {}
        self.first_stage_solution = None
        self.minimizing = None

    def scenario_creator(self, scenario_name, **kwargs):
        """
        Create the specified scenario
        """

        assert (
            scenario_name in self._scenario_probability
        ), f"Unknown scenario: {scenario_name}"

        # Create the concrete model object
        model = self._sp.create_subproblem(scenario_name)
        obj = find_objective(model)
        self.minimizing = obj.is_minimizing()

        # Add _mpisppy_probability
        model._mpisppy_probability = self._scenario_probability[scenario_name]

        # Add _nonant_vardata_list
        varlist = [
            self._sp.int_to_FirstStageVar[scenario_name][i]
            for i in sorted(self._sp.int_to_FirstStageVar[scenario_name].keys())
        ]
        model._nonant_vardata_list = mpisppy.utils.sputils.build_vardatalist(
            model, varlist
        )

        # Attach model to root node
        mpisppy.utils.sputils.attach_root_node(model, 0, varlist)

        return model

    def scenario_names_creator(self, num_scens, start=None):
        bundle_names = sorted(self._sp.bundles.keys())
        if num_scens is not None:
            assert (
                len(bundle_names) <= num_scens
            ), f"The stochastic program was defined with {len(bundle_names)=} bundles, but the user asked for {num_scens} bundles"
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
        return self.scenario_creator(sname, **sca)

    def scenario_denouement(self, rank, scenario_name, scenario):
        pass

    def custom_writer(self, wheel, cfg):
        def writer(client, file_name, scenario, bundling):
            root = scenario._mpisppy_node_list[0]
            assert root.name == "ROOT", f"Unexpected root name {root.name=}"
            root_nonants = np.fromiter(
                (pyo.value(var) for var in root.nonant_vardata_list), float
            )
            # comm = MPI.COMM_WORLD
            client.first_stage_solution = root_nonants

        wheel.write_first_stage_solution(
            "ignore.npy", first_stage_solution_writer=partial(writer, self)
        )

    def get_first_stage_solutions(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if rank > 0:
            if self.first_stage_solution:
                comm.send(self.first_stage_solution, dest=0, tag=0)
            else:
                comm.send([], dest=0, tag=0)
        elif rank == 0:
            results = {}
            while len(results) < size - 1:
                status = MPI.Status()
                tmp = comm.recv(source=MPI.ANY_SOURCE, tag=0, status=status)
                results[status.Get_source()] = tmp

            solutions = [v for v in results.values() if v]
            if self.first_stage_solution:
                solutions.append(self.first_stage_solution)
            return solutions


def mpisppy_generic_cylinders_main(module, options):
    import mpisppy.generic_cylinders as gc

    cfg = gc._parse_args(module)
    if options:
        for option, value in options.items():
            cfg[option] = value

    bundle_wrapper = None  # the default
    if gc._proper_bundles(cfg):
        import mpisppy.utils.proper_bundler as proper_bundler

        bundle_wrapper = proper_bundler.ProperBundler(module)
        bundle_wrapper.set_bunBFs(cfg)
        scenario_creator = bundle_wrapper.scenario_creator
        # The scenario creator is wrapped, so these kw_args will not go the original
        # creator (the kw_creator will keep the original args)
        scenario_creator_kwargs = bundle_wrapper.kw_creator(cfg)
    elif cfg.unpickle_scenarios_dir is not None:
        # So reading pickled scenarios cannot be composed with proper bundles
        scenario_creator = gc._read_pickled_scenario
        scenario_creator_kwargs = {"cfg": cfg}
    else:  # the most common case
        scenario_creator = module.scenario_creator
        scenario_creator_kwargs = module.kw_creator(cfg)

    assert hasattr(
        module, "scenario_denouement"
    ), "The model file must have a scenario_denouement function"
    scenario_denouement = module.scenario_denouement

    if cfg.pickle_bundles_dir is not None:
        global_comm = MPI.COMM_WORLD
        gc._write_bundles(
            module, cfg, scenario_creator, scenario_creator_kwargs, global_comm
        )
    elif cfg.pickle_scenarios_dir is not None:
        global_comm = MPI.COMM_WORLD
        gc._write_scenarios(
            module,
            cfg,
            scenario_creator,
            scenario_creator_kwargs,
            scenario_denouement,
            global_comm,
        )
    elif cfg.EF:
        gc._do_EF(
            module,
            cfg,
            scenario_creator,
            scenario_creator_kwargs,
            scenario_denouement,
            bundle_wrapper=bundle_wrapper,
        )
    else:
        if hasattr(gc, "do_decomp"):
            wheel = gc.do_decomp(
                module,
                cfg,
                scenario_creator,
                scenario_creator_kwargs,
                scenario_denouement,
                bundle_wrapper=bundle_wrapper,
            )
            return dict(
                BestInnerBound=wheel.BestInnerBound, BestOuterBound=wheel.BestOuterBound
            )

        else:
            gc._do_decomp(
                module,
                cfg,
                scenario_creator,
                scenario_creator_kwargs,
                scenario_denouement,
                bundle_wrapper=bundle_wrapper,
            )
            return dict()


def mpisppy_main(sp, options, argv):
    # Cache sys.argv
    old_argv = sys.argv
    # Clear sys.argv to force mpisppy to ignore it
    sys.argv = [sys.argv[0]] + argv

    guest = Forestlib_client(sp)
    results = mpisppy_generic_cylinders_main(guest, options)
    if guest.minimizing:
        results = dict(
            lower_bound=results.get("BestOuterBound", None),
            best_value=results.get("BestInnerBound", None),
        )
    else:
        results = dict(
            upper_bound=results.get("BestInnerBound", None),
            best_value=results.get("BestOuterBound", None),
        )
    results["first_stage_solutions"] = guest.get_first_stage_solutions()

    # Reset sys.argv
    sys.argv = old_argv
    return results


class ProgressiveHedgingSolver_MPISPPY(object):

    def __init__(self):
        if mpisppy_available:
            comm = MPI.COMM_WORLD
            self.mpi_rank = comm.Get_rank()
        self.rho = {}
        # self.cached_model_generation = True
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
        self.mpisppy_options = []

    def set_options(
        self,
        *,
        rho=None,
        # cached_model_generation=None,
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
        # solution_manager=None,
        rho_updates=False,
        default_rho=None,
        mpisppy_options=None,
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
        # if cached_model_generation is not None:
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
        # if solution_manager is not None:
        #    self.solution_manager = solution_manager
        if mpisppy_options:
            self.mpisppy_options = mpisppy_options

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

        if self.mpi_rank == 0 and logger.isEnabledFor(logging.DEBUG):
            print("Solver Configuration")
            # print(f"  cached_model_generation    {self.cached_model_generation}")
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
        if self.mpi_rank == 0:
            if self.solutions is None:
                self.solutions = solnpool.PoolManager()
            if self.finalize_all_xbar:
                sp_metadata = self.solutions.add_pool(
                    "PH Iterations", policy="keep_all"
                )
            else:
                sp_metadata = self.solutions.add_pool(
                    "PH Iterations", policy="keep_latest"
                )
            sp_metadata.solver = "PH Iteration Results"
            sp_metadata.solver_options = dict(
                # cached_model_generation=self.cached_model_generation,
                max_iterations=self.max_iterations,
                time_limit=self.time_limit,
                convergence_tolerance=self.convergence_tolerance,
                normalize_convergence_norm=self.normalize_convergence_norm,
                solver_name=self.solver_name,
                solver_options=self.solver_options,
            )

        if self.mpi_rank == 0:
            logger.info("ProgressiveHedgingSolver_MPISPPY - START")

        options = {
            "solver_name": self.solver_name,
        }
        if self.default_rho:
            options["default_rho"] = self.default_rho
        if self.max_iterations:
            options["max_iterations"] = self.max_iterations
        if self.time_limit:
            options["time_limit"] = self.time_limit
        if self.solver_options:
            options["solver_options"] = self.solver_options
        if logger.isEnabledFor(logging.INFO):
            options["display_progress"] = True
        if logger.isEnabledFor(logging.VERBOSE):
            options["verbose"] = True
        options["xhatxbar"] = True
        options["num_scens"] = len(sp.bundles)

        results = mpisppy_main(sp, options, self.mpisppy_options)

        if self.mpi_rank == 0:
            end_time = datetime.datetime.now()

            sp_metadata = self.solutions.metadata
            # sp_metadata.iterations = iteration
            sp_metadata.termination_condition = "ok"
            sp_metadata.start_time = str(start_time)
            if results["lower_bound"]:
                sp_metadata.lower_bound = float(results["lower_bound"])
            elif results["upper_bound"]:
                sp_metadata.upper_bound = float(results["upper_bound"])

            logger.info("")
            logger.info("-" * 70)
            logger.info("ProgressiveHedgingSolver_MPISPPY - FINALIZING")

            for soln in results["first_stage_solutions"]:
                args = dict(sp=sp, xbar=soln)
                if results["best_value"]:
                    args["objective"] = solnpool.Objective(
                        value=float(results["best_value"])
                    )
                self.archive_solution(**args)

            sp_metadata.end_time = str(end_time)
            sp_metadata.time_elapsed = str(end_time - start_time)

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

    def archive_solution(self, *, sp, xbar, w=None, **kwds):
        w = {} if w is None else w
        variables = [
            solnpool.Variable(
                value=float(val),
                index=i,
                name=sp.get_variable_name(i),
                suffix=munch.Munch(w={k: v[i] for k, v in w.items()}),
            )
            for i, val in enumerate(xbar)
        ]
        return self.solutions.add(variables=variables, **kwds)
