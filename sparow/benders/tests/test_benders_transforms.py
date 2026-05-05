import pytest
import pyomo.environ as pyo

from sparow.sp.examples import (
    simple_absolute_value,
    feasibility_included_absolute_value,
    absolute_value_testing_version,
    adjustable_absolute_value,
)

from pyomo.common.dependencies import attempt_import

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized

from sparow.benders import BendersSolver
from or_topas.util.pyomo_utils import split_expr

import pyomo.opt
from pyomo.common import unittest
from pyomo.core.expr.compare import compare_expressions
from pyomo.repn.standard_repn import generate_standard_repn

open_source_solver = set(pyomo.opt.check_available_solvers("highs"))
if len(open_source_solver) == 0:
    open_source_solver = set(pyomo.opt.check_available_solvers("glpk"))
# solvers = set(pyomo.opt.check_available_solvers("gurobi")) | open_source_solver
solvers = set(pyomo.opt.check_available_solvers("gurobi"))


class TestBendersTransforms:

    def test_BendersSolverSetup(self):
        solver = BendersSolver()
        solver.set_options(solver=solver, subproblem_solver=solver)

    def test_transform_to_subproblem_model_domain_changes(self):
        solver = open_source_solver

        solver = BendersSolver()
        solver.set_options(solver=solver, subproblem_solver=solver)
        domains_to_check = [pyo.Reals, pyo.NonNegativeIntegers, pyo.NonNegativeReals]
        for d in domains_to_check:
            app = absolute_value_testing_version()
            sp = app.sp
            b = next(iter(sp.bundles))

            m = BendersSolver._transform_to_subproblem_model(
                sp_lower=sp,
                b=b,
                default_domain=d,
                remove_first_stage_only_cons=False,
                weight_obj_by_prob=True,
                remove_first_stage_objective_terms=False,
            )
            # variables are under m.s[None,1]
            assert (
                m.s[None, 1].x.domain is d
            ), f"Expected first_stage_variable to be set to domain: {d}"
            assert all(
                m.s[None, 1].y[i].domain is pyo.Reals
                for i in m.s[None, 1].y.index_set()
            ), f"Expected all second_stage_variables to stay as pyo.Reals"

    def test_transform_to_subproblem_constraint_updates(self):
        solver = open_source_solver

        solver = BendersSolver()
        solver.set_options(solver=solver, subproblem_solver=solver)
        domains_to_check = [pyo.Reals, pyo.NonNegativeIntegers, pyo.NonNegativeReals]
        removal_options = [True, False]
        for r_option in removal_options:
            app = absolute_value_testing_version()
            sp = app.sp
            b = next(iter(sp.bundles))
            m = BendersSolver._transform_to_subproblem_model(
                sp_lower=sp,
                b=b,
                default_domain=pyo.Reals,
                remove_first_stage_only_cons=r_option,
                weight_obj_by_prob=True,
                remove_first_stage_objective_terms=True,
            )
            cons_list = [
                c
                for c in m.component_data_objects(
                    pyo.Constraint, descend_into=True, active=True
                )
            ]
            if r_option:
                assert m.s[None, 1].vertex_cons.active
                assert not m.s[None, 1].x_lower.active
                assert not m.s[None, 1].x_upper.active
                assert len(cons_list) == 1
            else:
                assert m.s[None, 1].vertex_cons.active
                assert m.s[None, 1].x_lower.active
                assert m.s[None, 1].x_upper.active
                assert len(cons_list) == 3

    # multiple objective question
    def Xtest_transform_to_subproblem_model_update_objective_1(self):
        solver = open_source_solver
        solver = BendersSolver()
        solver.set_options(solver=solver, subproblem_solver=solver)

        model_data = {
            "scenarios": [
                {"ID": 1, "LB": None, "UB": None},
            ],
        }
        c_values = [0, 1, 2]
        removal_options = [True, False]
        for c in c_values:
            for r_option in removal_options:
                app_data = dict(a=0, c=c, L=1, R=1)
                app = adjustable_absolute_value(
                    local_app_data=app_data, local_model_data=model_data
                )
                sp = app.sp
                b = next(iter(sp.bundles))
                m = BendersSolver._transform_to_subproblem_model(
                    sp_lower=sp,
                    b=b,
                    default_domain=pyo.Reals,
                    remove_first_stage_only_cons=False,
                    weight_obj_by_prob=True,
                    remove_first_stage_objective_terms=r_option,
                )
                # why are there two objectives here, I don't understand what is going on in sp
                # all the variables are under m.s[None,1], why is the objective under m not m.s[None,1]
                m.pprint()
                print(
                    f"Obj transform update: {str(m.s[None,1].obj.expr)=}, {m.s[None,1].obj.active}"
                )
                print(f"Obj transform update: {str(m.obj.expr)=}, {m.obj.active}")
                split = split_expr(
                    m.s[None, 1].obj.expr, [m.s[None, 1].x], allow_iterables=True
                )
                assert split.constant == pytest.approx(0)
                assert compare_expressions(
                    split.not_in_set,
                    app_data["R"] * m.s[None, 1].y["Right"]
                    + app_data["L"] * m.s[None, 1].y["Left"],
                )
                # test_expr = app_data["R"] * m.s[None,1].y["Right"] + app_data["L"] * m.s[None,1].y["Left"]
                # if not r_option:
                #     test_expr += app_data["R"] * m.s[None,1].x
                # assert compare_expressions(m.s[None,1].obj.expr, test_expr)
                print(f"{c=},{r_option=},{split.in_set=}")
                if r_option:
                    assert split.in_set == pytest.approx(0)

    def test_transform_to_subproblem_model_update_objective_1(self):
        solver = open_source_solver
        solver = BendersSolver()
        solver.set_options(solver=solver, subproblem_solver=solver)

        model_data = {
            "scenarios": [
                {"ID": 1, "LB": None, "UB": None},
            ],
        }
        c_values = [1, 2]
        removal_options = [True, False]
        for c in c_values:
            for r_option in removal_options:
                app_data = dict(a=0, c=c, L=1, R=1)
                app = adjustable_absolute_value(
                    local_app_data=app_data, local_model_data=model_data
                )
                sp = app.sp
                b = next(iter(sp.bundles))
                m = BendersSolver._transform_to_subproblem_model(
                    sp_lower=sp,
                    b=b,
                    default_domain=pyo.Reals,
                    remove_first_stage_only_cons=False,
                    weight_obj_by_prob=True,
                    remove_first_stage_objective_terms=r_option,
                )
                split = split_expr(m.obj.expr, [m.s[None, 1].x], allow_iterables=True)
                assert split.constant == pytest.approx(0)
                assert compare_expressions(
                    split.not_in_set,
                    app_data["R"] * m.s[None, 1].y["Right"]
                    + app_data["L"] * m.s[None, 1].y["Left"],
                )
                if r_option:
                    assert split.in_set == pytest.approx(0)
                else:
                    assert compare_expressions(split.in_set, c * m.s[None, 1].x)

    def Xtest_simple(self, mip_solver):
        app = simple_newsvendor()
        solver = BendersSolver()
        solver.set_options(solver=mip_solver, subproblem_solver=mip_solver)

        default_lower_eta = -1_000
        eta_bounds_map = {s: (default_lower_eta, None) for s in app.sp.bundles}
        results = solver.solve_in_dev(app.sp, eta_bounds_map)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])
