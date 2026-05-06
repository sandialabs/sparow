import pytest
import pyomo.environ as pyo

from sparow.sp.examples import (
    simple_absolute_value,
    feasibility_included_absolute_value,
    absolute_value_testing_version,
    adjustable_absolute_value,
    simple_newsvendor,
)

from pyomo.common.dependencies import attempt_import

parameterized, param_available = attempt_import("parameterized")
if not param_available:
    raise unittest.SkipTest("Parameterized is not available.")
parameterized = parameterized.parameterized

from sparow.benders import BendersSolver
from sparow.ef import ExtensiveFormSolver
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

    def test_simple_ef_check(self):
        mip_solver = next(iter(open_source_solver))
        app = simple_absolute_value()
        solver = ExtensiveFormSolver()
        solver.set_options(solver=mip_solver)
        results = solver.solve(app.sp)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

    def test_create_sp_upper(self):
        solver_name = next(iter(open_source_solver))

        app = simple_absolute_value()
        sp_lower = app.sp
        sp_upper = BendersSolver._create_sp_upper(sp_lower=sp_lower)
        test_a_val = 1
        sp_upper.app_data["a"] = test_a_val

        # default behavior check for sp_lower
        solver = ExtensiveFormSolver()
        solver.set_options(solver=solver_name)
        results = solver.solve(sp_lower)
        results_dict = results.to_dict()
        soln = next(iter(results_dict["solutions"].values()))

        obj_val = soln["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x = soln["variables"][0]["value"]
        assert x == pytest.approx(app.solution_values["x"])

        # behavior check for sp_upper
        solver2 = ExtensiveFormSolver()
        solver2.set_options(solver=solver_name)
        results2 = solver2.solve(sp_upper)
        results_dict2 = results2.to_dict()
        soln2 = next(iter(results_dict2["solutions"].values()))

        obj_val2 = soln2["objectives"][0]["value"]
        assert obj_val == pytest.approx(app.objective_value)
        assert app.unique_solution
        x2 = soln2["variables"][0]["value"]
        assert x2 == pytest.approx(test_a_val)

    def test_transform_to_subproblem_constraint_updates(self):

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

    # two objectives point applies here too
    # why is obj on m and m.s[None,b] but only active on one
    def Xtest_transform_to_subproblem_prob_weight_obj_weird(self):

        solver = BendersSolver()
        solver.set_options(solver=solver, subproblem_solver=solver)

        # obj_options = [True, False]
        obj_options = [True]
        for o_option in obj_options:
            app = simple_newsvendor()
            sp = app.sp
            for b in sp.bundles:
                m = BendersSolver._transform_to_subproblem_model(
                    sp_lower=sp,
                    b=b,
                    default_domain=pyo.Reals,
                    remove_first_stage_only_cons=False,
                    weight_obj_by_prob=o_option,
                    remove_first_stage_objective_terms=True,
                )
                m.pprint()
                print(m.obj.expr)
                print(m.s)
                print(list(m.s.keys()))
                # print(m.s[(None,b)])
                # print(sp.int_to_FirstStageVar[b][0].name)

                # ask bill what is going on with erroring on this access
                # print(m.s[None,b].y)
                factor = 0.2 if o_option else 1
                generate_standard_repn(m.obj.expr)
                # assert compare_expressions(
                #     m.obj.expr,
                #     factor * m.s[None,b].y,
                # )
                # m.pprint()
                # print(f"{sp.bundles[b].probability=}, {b=}")

                # b = next(iter(sp.bundles))
                # m = BendersSolver._transform_to_subproblem_model(
                #     sp_lower=sp,
                #     b=b,
                #     default_domain=pyo.Reals,
                #     remove_first_stage_only_cons=False,
                #     weight_obj_by_prob=o_option,
                #     remove_first_stage_objective_terms=True,
                # )
                # cons_list = [
                #     c
                #     for c in m.component_data_objects(
                #         pyo.Constraint, descend_into=True, active=True
                #     )
                # ]
                # if r_option:
                #     assert m.s[None, 1].vertex_cons.active
                #     assert not m.s[None, 1].x_lower.active
                #     assert not m.s[None, 1].x_upper.active
                #     assert len(cons_list) == 1
                # else:
                #     assert m.s[None, 1].vertex_cons.active
                #     assert m.s[None, 1].x_lower.active
                #     assert m.s[None, 1].x_upper.active
                #     assert len(cons_list) == 3

    def test_transform_to_subproblem_prob_weight_obj(self):

        solver = BendersSolver()
        solver.set_options(solver=solver, subproblem_solver=solver)

        obj_options = [True, False]
        # obj_options = [True]
        for o_option in obj_options:
            app = simple_newsvendor()
            sp = app.sp
            for b in sp.bundles:
                m = BendersSolver._transform_to_subproblem_model(
                    sp_lower=sp,
                    b=b,
                    default_domain=pyo.Reals,
                    remove_first_stage_only_cons=False,
                    weight_obj_by_prob=o_option,
                    remove_first_stage_objective_terms=True,
                )
                factor = 0.2 if o_option else 1
                repn = generate_standard_repn(m.obj.expr)
                assert repn.linear_coefs == pytest.approx([factor])

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
