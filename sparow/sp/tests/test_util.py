from sparow.sp.examples import simple_newsvendor
from sparow.sp.util import SparowSolution
from sparow.sp.util import SparowPoolManager
from or_topas.solnpool.solnpool import PoolPolicy


def test_sparow_solutions():
    app = simple_newsvendor() # using model with known solution
    pm = SparowPoolManager()  # create pool manager
    pm.add_pool(name="pool_1", policy=PoolPolicy.keep_all)
    # create sparow solution with known optimal obj/val (i.e., without solving):
    variables=[
            pm.create_variable(name=key, value=val)
            for key, val in app.solution_values.items()
        ] 
    objective_list=[pm.create_objective(value=app.objective_value)] 
    # add sparow solution to pool manager:
    sparow_soln_ID = pm.add(
        variables=variables, objectives=objective_list
    )

    assert sparow_soln_ID is not None  # ensure add method is returning soln ID

    assert pm.get_pool_dicts() == {  # ensure sparow solution and pool manager have expected structure/keys/values
        "pool_1": {
            "metadata": {
                "as_solution_source": "sparow.sp.util._sparow_as_solution",
                "context_name": "pool_1",
                "policy": "keep_all",
            },
            "pool_config": {},
            "solutions": {0: pm[sparow_soln_ID].to_dict()},
        }
    }
