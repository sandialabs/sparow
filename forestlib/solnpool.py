import dataclasses
import json
import munch

nan = float("nan")


def _custom_dict_factory(data):
    return {k: v.toDict() if type(v) is munch.Munch else v for k, v in data}


@dataclasses.dataclass
class Variable:
    _: dataclasses.KW_ONLY
    value: float = nan
    fixed: bool = False
    name: str = None
    repn = None
    index: int = None
    discrete: bool = False
    suffix: munch.Munch = dataclasses.field(default_factory=munch.Munch)

    def to_dict(self):
        return dataclasses.asdict(self, dict_factory=custom_dict_factory)


@dataclasses.dataclass
class Objective:
    _: dataclasses.KW_ONLY
    value: float = nan
    name: str = None
    suffix: munch.Munch = dataclasses.field(default_factory=munch.Munch)

    def to_dict(self):
        return dataclasses.asdict(self, dict_factory=custom_dict_factory)


class Solution:

    _id_counter = 0

    def __init__(self, *, variables=None, objectives=None, **kwds):
        self.id = self._id_counter
        self._id_counter += 1

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
            for o in objectives:
                if o.name is not None:
                    self.str_to_objective[o.name] = o

        self.suffix = munch.Munch(**kwds)

    def variable(self, index):
        if type(index) is int:
            return self.int_to_variable[index]
        else:
            return self.str_to_variable[index]

    def variables(self):
        return self._variables

    def objective(self, index):
        if type(index) is int:
            return self.int_to_objective[index]
        else:
            return self.str_to_objective[index]

    def objectives(self):
        return self._objectives

    def to_dict(self):
        return dict(
            variables=[v.to_dict() for v in self.variables()],
            objectives=[v.to_dict() for o in self.objectives()],
        )


class SolutionPool:

    def __init__(self):
        self._context = "none"
        self.solutions = {"none":{}}

    @property
    def context(self):
        return self._context

    def set_context(self, context):
        if context not in self.solutions:
            self.solutions[context] = {}
        self._context = context

    def add(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            assert type(args[0]) is Solution, "Expected a single solution"
            soln = args[0]
        else:
            soln = Solution(*args, **kwargs)
        self.solutions[self._context][soln.id] = soln

    def to_dict(self):
        return {k:{i:soln.to_dict() for i,soln in v.items()} for k,v in self.solutions.items()}

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
            self.solutions = data.solutions

