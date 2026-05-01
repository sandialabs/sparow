import pytest
import pyomo.environ as pyo
from sparow.sp import stochastic_program
from sparow.sp.util import relax_second_stage
from sparow.ef import ExtensiveFormSolver

from pyomo.opt import check_available_solvers

highs_available = len(check_available_solvers("highs")) == 1


@pytest.fixture
def sp0():
    def builder(*args, **kwargs):
        M = pyo.ConcreteModel()
        M.x = pyo.Var(domain=pyo.Binary)
        M.y = pyo.Var(domain=pyo.Binary)
        M.o = pyo.Objective(expr=M.x)
        return M

    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_model(model_builder=builder)
    sp.initialize_model(model_data=dict(scenarios=[dict(ID=1)]))
    return sp


@pytest.fixture
def sp1():
    def builder(*args, **kwargs):
        M = pyo.ConcreteModel()
        x_ind = [1, 2, 3, 4]
        y_ind = ["a", "b", "c", "d"]
        M.x = pyo.Var(x_ind, domain=pyo.Binary)
        M.y = pyo.Var(y_ind, domain=pyo.Binary)
        M.o = pyo.Objective(expr=sum(M.x))
        return M

    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_model(model_builder=builder)
    sp.initialize_model(model_data=dict(scenarios=[dict(ID=1)]))
    return sp


@pytest.fixture
def sp2():
    def builder(*args, **kwargs):
        M = pyo.ConcreteModel()
        M.x = pyo.Var(domain=pyo.Binary)
        M.y = pyo.Var(domain=pyo.Binary)
        M.o = pyo.Objective(expr=M.x)
        return M

    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_model(model_builder=builder)
    sp.initialize_model(
        model_data={
            "scenarios": [
                {"ID": 1},
                {"ID": 2},
            ]
        }
    )
    return sp


@pytest.fixture
def sp3():
    def builder(*args, **kwargs):
        M = pyo.ConcreteModel()
        M.x = pyo.Var(domain=pyo.Binary)
        M.y = pyo.Var(domain=pyo.Binary)
        M.con = pyo.Constraint(expr=2 * M.x + 2 * M.y <= 3)
        M.o = pyo.Objective(expr=-M.x - M.y)
        return M

    sp = stochastic_program(first_stage_variables=["x"])
    sp.initialize_model(model_builder=builder)
    sp.initialize_model(
        model_data={
            "scenarios": [
                {"ID": 1},
                {"ID": 2},
            ]
        }
    )
    sp.initialize_bundles(scheme="single_bundle")
    return sp


class TestRSS(object):
    """
    Test relax second stage utility function
    """

    def test_second_stage_vars_relaxed(self, sp0):
        sp0.add_transformation(relax_second_stage)
        M = sp0.create_subproblem("1")

        for b in M.s.index_set():
            key = b
            assert M.s[key].x.domain == pyo.Binary
            assert M.s[key].y.domain == pyo.Reals

    def test_indexed_vars(self, sp1):
        sp1.add_transformation(relax_second_stage)
        M = sp1.create_subproblem("1")

        for b in M.s.index_set():
            key = b
            for ind in M.s[key].x.index_set():
                assert M.s[key].x[ind].domain == pyo.Binary
            for ind in M.s[key].y.index_set():
                assert M.s[key].y[ind].domain == pyo.Reals

    def test_relax_dict(self, sp2):
        rd = {1: True, 2: False}
        sp2.add_transformation(relax_second_stage, relax_dict=rd)
        M = sp2.create_EF()

        for b in M.s.index_set():
            key = b[-1] if isinstance(b, tuple) else b

            if key == 1:
                assert M.s[b].x.domain == pyo.Binary
                assert M.s[b].y.domain == pyo.Reals

            if key == 2:
                assert M.s[b].x.domain == pyo.Binary
                assert M.s[b].y.domain == pyo.Binary

    @pytest.mark.skipif(not highs_available, reason="highs not installed")
    def test_solve_model_no_relax(self, sp3):
        sp3.initialize_bundles(scheme="single_bundle")
        b = next(iter(sp3.bundles))
        M = sp3.create_subproblem(b, compact_repn=False)
        results = sp3.solve(M, solver="highs", tee=True)

        for b in M.s.index_set():
            key = b[-1] if isinstance(b, tuple) else b
            if key == 1:
                assert pyo.value(M.s[b].x) == 0.0
                assert pyo.value(M.s[b].y) == 1.0

            if key == 2:
                assert pyo.value(M.s[b].x) == 0.0
                assert pyo.value(M.s[b].y) == 1.0

    @pytest.mark.skipif(not highs_available, reason="highs not installed")
    def test_solve_model_relax(self, sp3):
        rd = {1: True, 2: True}
        sp3.add_transformation(relax_second_stage, relax_dict=rd)
        sp3.initialize_bundles(scheme="single_bundle")
        b = next(iter(sp3.bundles))
        M = sp3.create_subproblem(b, compact_repn=False)
        results = sp3.solve(M, solver="highs", tee=True)

        for b in M.s.index_set():
            key = b[-1] if isinstance(b, tuple) else b
            if key == 1:
                assert pyo.value(M.s[b].x) == 1.0
                assert pyo.value(M.s[b].y) == 0.5

            if key == 2:
                assert pyo.value(M.s[b].x) == 1.0
                assert pyo.value(M.s[b].y) == 0.5
