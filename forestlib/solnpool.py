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

    _id_counter = 0

    def __init__(self, *, variables=None, objectives=None, **kwds):
        self.id = self._id_counter
        Solution._id_counter += 1

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


class SolutionPool:

    def __init__(self):
        self._context_name = "none"
        self._solutions = {}
        self.set_context(self._context_name)

    @property
    def metadata(self):
        return self._solutions[self._context_name].context

    @property
    def solutions(self):
        return self._solutions[self._context_name].solutions.values()

    @property
    def last_solution(self):
        context = self._solutions[self._context_name]
        index = next(reversed(context.solutions.keys()))
        return context.solutions[index]

    def __iter__(self):
        for soln in self._solutions[self._context_name].solutions.values():
            yield soln

    def __len__(self):
        return len(self._solutions[self._context_name].solutions)

    def __getitem__(self, soln_id, context_name=None):
        if context_name is None:
            context_name = self._context_name
        return self._solutions[context_name].solutions[soln_id]

    def set_context(self, name, *, hash_variables=True):
        if name not in self._solutions:
            self._solutions[name] = MyMunch(
                context=MyMunch(context_name=name),
                solutions={},
                hash_variables=hash_variables,
                unique_solutions=set(),
            )
        self._context_name = name
        return self.metadata

    def add(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            assert type(args[0]) is Solution, "Expected a single solution"
            soln = args[0]
        else:
            soln = Solution(*args, **kwargs)
        #
        # Return None if hash_variables is True and the solution has already been added to the pool
        #
        context = self._solutions[self._context_name]
        if context.hash_variables:
            tuple_repn = soln.tuple_repn()
            if tuple_repn in context.unique_solutions:
                return None

        assert (
            soln.id not in context.solutions
        ), f"Solution id {soln.id} already in solution pool context '{self._context_name}'"
        context.solutions[soln.id] = soln
        return soln.id

    def to_dict(self):
        return _to_dict(self._solutions)

    def write(self, json_filename, indent=None):
        with open(json_filename, "w") as OUTPUT:
            json.dump(OUTPUT, self.to_dict(), indent=indent)

    def read(self, json_filename):
        assert os.path.exists(
            json_filename
        ), f"ERROR: file '{json_filename}' does not exist!"
        with open(json_filename, "r") as INPUT:
            try:
                data = json.load(INPUT)
            except ValueError as e:
                raise ValueError(f"Invalid JSON in file '{json_filename}': {e}")
            self._solutions = data.solutions
