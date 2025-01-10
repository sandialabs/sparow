import collections
import dataclasses
import json
import munch

nan = float("nan")


class MyMunch(munch.Munch):

    to_dict = munch.Munch.toDict


def _to_dict(x):
    xtype = type(x)
    if xtype in [float, int, complex, str, list, bool] or x is None:
        return x
    elif xtype in [tuple, set, frozenset]:
        return list(x)
    elif xtype in [dict, munch.Munch, MyMunch]:
        return {k: _to_dict(v) for k, v in x.items()}
    else:
        return x.to_dict()


def _custom_dict_factory(data):
    return {k: _to_dict(v) for k, v in data}


@dataclasses.dataclass
class Variable:
    _: dataclasses.KW_ONLY
    value: float = nan
    fixed: bool = False
    name: str = None
    repn = None
    index: int = None
    discrete: bool = False
    suffix: MyMunch = dataclasses.field(default_factory=MyMunch)

    def to_dict(self):
        return dataclasses.asdict(self, dict_factory=_custom_dict_factory)


@dataclasses.dataclass
class Objective:
    _: dataclasses.KW_ONLY
    value: float = nan
    name: str = None
    suffix: MyMunch = dataclasses.field(default_factory=MyMunch)

    def to_dict(self):
        return dataclasses.asdict(self, dict_factory=_custom_dict_factory)


class Solution:

    def __init__(self, *, variables=None, objectives=None, **kwds):
        self.id = None

        self._variables = []
        self.int_to_variable = {}
        self.str_to_variable = {}
        if variables is not None:
            self._variables = variables
            for v in variables:
                if v.index is not None:
                    self.int_to_variable[v.index] = v
                if v.name is not None:
                    self.str_to_variable[v.name] = v

        self._objectives = []
        self.str_to_objective = {}
        if objectives is not None:
            self._objectives = objectives
        elif "objective" in kwds:
            self._objectives = [kwds.pop("objective")]
        for o in self._objectives:
            self.str_to_objective[o.name] = o

        if "suffix" in kwds:
            self.suffix = MyMunch(kwds.pop("suffix"))
        else:
            self.suffix = MyMunch(**kwds)

    def variable(self, index):
        if type(index) is int:
            return self.int_to_variable[index]
        else:
            return self.str_to_variable[index]

    def variables(self):
        return self._variables

    def tuple_repn(self):
        if len(self.int_to_variable) == len(self._variables):
            return tuple(
                tuple([k, var.value]) for k, var in self.int_to_variable.items()
            )
        elif len(self.str_to_variable) == len(self._variables):
            return tuple(
                tuple([k, var.value]) for k, var in self.str_to_variable.items()
            )
        else:
            return tuple(tuple([k, var.value]) for k, var in enumerate(self._variables))

    def objective(self, index=None):
        if type(index) is int:
            return self.int_to_objective[index]
        else:
            return self.str_to_objective[index]

    def objectives(self):
        return self._objectives

    def to_dict(self):
        return dict(
            id=self.id,
            variables=[v.to_dict() for v in self.variables()],
            objectives=[o.to_dict() for o in self.objectives()],
            suffix=self.suffix.to_dict(),
        )


class SolutionPoolBase:

    _id_counter = 0

    def __init__(self, name=None):
        self.metadata = MyMunch(context_name=name)
        self._solutions = {}

    @property
    def solutions(self):
        return self._solutions.values()

    @property
    def last_solution(self):
        index = next(reversed(self._solutions.keys()))
        return self._solutions[index]

    def __iter__(self):
        for soln in self._solutions.values():
            yield soln

    def __len__(self):
        return len(self._solutions)

    def __getitem__(self, soln_id):
        return self._solutions[soln_id]

    def _as_solution(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            assert type(args[0]) is Solution, "Expected a single solution"
            return args[0]
        return Solution(*args, **kwargs)


class SolutionPool_KeepAll(SolutionPoolBase):

    def __init__(self, name=None):
        super().__init__(name)

    def add(self, *args, **kwargs):
        soln = self._as_solution(*args, **kwargs)
        #
        soln.id = SolutionPoolBase._id_counter
        SolutionPoolBase._id_counter += 1
        assert (
            soln.id not in self._solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        #
        self._solutions[soln.id] = soln
        return soln.id

    def to_dict(self):
        return dict(
            metadata=_to_dict(self.metadata),
            solutions=_to_dict(self._solutions),
            pool_policy=dict(policy="keep_all"),
        )


class SolutionPool_KeepLatest(SolutionPoolBase):

    def __init__(self, name=None, *, max_pool_size=1):
        super().__init__(name)
        self.max_pool_size = max_pool_size
        self.int_deque = collections.deque()

    def add(self, *args, **kwargs):
        soln = self._as_solution(*args, **kwargs)
        #
        soln.id = SolutionPoolBase._id_counter
        SolutionPoolBase._id_counter += 1
        assert (
            soln.id not in self._solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        #
        self.int_deque.append(soln.id)
        if len(self.int_deque) > self.max_pool_size:
            index = self.int_deque.popleft()
            del self._solutions[index]
        #
        self._solutions[soln.id] = soln
        return soln.id

    def to_dict(self):
        return dict(
            metadata=_to_dict(self.metadata),
            solutions=_to_dict(self._solutions),
            pool_policy=dict(policy="latest", max_pool_size=self.max_pool_size),
        )


class SolutionPool_KeepLatestUnique(SolutionPoolBase):

    def __init__(self, name=None, *, max_pool_size=None):
        super().__init__(name)
        self.max_pool_size = max_pool_size
        self.int_deque = collections.deque()
        self.unique_solutions = set()

    def add(self, *args, **kwargs):
        soln = self._as_solution(*args, **kwargs)
        #
        # Return None if the solution has already been added to the pool
        #
        tuple_repn = soln.tuple_repn()
        if tuple_repn in self.unique_solutions:
            return None
        self.unique_solutions.add(tuple_repn)
        #
        soln.id = SolutionPoolBase._id_counter
        SolutionPoolBase._id_counter += 1
        assert (
            soln.id not in self._solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        #
        self.int_deque.append(soln.id)
        if len(self.int_deque) > self.max_pool_size:
            index = self.int_deque.popleft()
            del self._solutions[index]
        #
        self._solutions[soln.id] = soln
        return soln.id

    def to_dict(self):
        return dict(
            metadata=_to_dict(self.metadata),
            solutions=_to_dict(self._solutions),
            pool_config=dict(policy="latest_unique", max_pool_size=self.max_pool_size),
        )


# TODO - setup heap logic here


class SolutionPool_KeepBest(SolutionPoolBase):

    def __init__(
        self,
        name=None,
        *,
        max_pool_size=None,
        objective=None,
        abs_tolerance=1e-7,
        rel_tolerance=1e-7,
    ):
        super().__init__(name)
        self.max_pool_size = max_pool_size
        self.objective = objective
        self.abs_tolerance = abs_tolerance
        self.rel_tolerance = rel_tolerance
        self.int_deque = collections.deque()
        self.unique_solutions = set()

    def add(self, *args, **kwargs):
        soln = self._as_solution(*args, **kwargs)
        #
        # Return None if the solution has already been added to the pool
        #
        tuple_repn = soln.tuple_repn()
        if tuple_repn in self.unique_solutions:
            return None
        self.unique_solutions.add(tuple_repn)
        #
        soln.id = SolutionPoolBase._id_counter
        SolutionPoolBase._id_counter += 1
        assert (
            soln.id not in self._solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        #
        self.int_deque.append(soln.id)
        if self.max_pool_size is not None and len(self.int_deque) > self.max_pool_size:
            index = self.int_deque.popleft()
            del self._solutions[index]
        #
        self._solutions[soln.id] = soln
        return soln.id

    def to_dict(self):
        return dict(
            metadata=_to_dict(self.metadata),
            solutions=_to_dict(self._solutions),
            pool_config=dict(
                policy="best",
                max_pool_size=self.max_pool_size,
                objective=self.objective,
                abs_tolerance=self.abs_tolerance,
                rel_tolerance=self.rel_tolerance,
            ),
        )


class SolutionManager:

    def __init__(self):
        self._name = None
        self._pool = {}
        self.add_pool(self._name)

    @property
    def pool(self):
        assert self._name in self._pool, f"Unknown pool '{self._name}'"
        return self._pool[self._name]

    @property
    def metadata(self):
        return self.pool.metadata

    @property
    def solutions(self):
        return self.pool.solutions.values()

    @property
    def last_solution(self):
        return self.pool.last_solution

    def __iter__(self):
        for soln in self.pool.solutions:
            yield soln

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, soln_id, name=None):
        if name is None:
            name = self._name
        return self._pool[name][soln_id]

    def add_pool(self, name, *, policy="keep_best", **kwds):
        if name not in self._pool:
            # Delete the 'None' pool if it isn't being used
            if name is not None and None in self._pool and len(self._pool[None]) == 0:
                del self._pool[None]

            if policy == "keep_all":
                self._pool[name] = SolutionPool_KeepAll(name=name)
            elif policy == "keep_best":
                self._pool[name] = SolutionPool_KeepBest(name=name, **kwds)
            elif policy == "keep_latest":
                self._pool[name] = SolutionPool_KeepLatest(name=name, **kwds)
            elif policy == "keep_latest_unique":
                self._pool[name] = SolutionPool_KeepLatestUnique(name=name, **kwds)
            else:
                raise ValueError(f"Unknown pool policy: {policy}")
        self._name = name
        return self.metadata

    def set_pool(self, name):
        assert name in self._pool, f"Unknown pool '{name}'"
        self._name = name
        return self.metadata

    def add(self, *args, **kwargs):
        return self.pool.add(*args, **kwargs)

    def to_dict(self):
        return {k: v.to_dict() for k, v in self._pool.items()}

    def write(self, json_filename, indent=None, sort_keys=True):
        with open(json_filename, "w") as OUTPUT:
            json.dump(self.to_dict(), OUTPUT, indent=indent, sort_keys=sort_keys)

    def read(self, json_filename):
        assert os.path.exists(
            json_filename
        ), f"ERROR: file '{json_filename}' does not exist!"
        with open(json_filename, "r") as INPUT:
            try:
                data = json.load(INPUT)
            except ValueError as e:
                raise ValueError(f"Invalid JSON in file '{json_filename}': {e}")
            self._pool = data.solutions
