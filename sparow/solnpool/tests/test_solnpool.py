from sparow.sp.examples import simple_newsvendor
from sparow import solnpool


def test_sparow_solutions():
    app = simple_newsvendor()  # using model with known solution
    solutions = solnpool.SparowPoolManager()  # create pool manager
    solutions.add_pool(
        name="pool_1", policy=solnpool.PoolPolicy.keep_all
    )  # create pool that keeps all solutions
    # create sparow solution with known optimal obj/val (i.e., without solving):
    variables = [
        solnpool.create_variable(name=key, value=val)
        for key, val in app.solution_values.items()
    ]
    objectives = [solnpool.create_objective(value=app.objective_value)]
    # add sparow solution to pool manager:
    sparow_soln_ID = solutions.add(variables=variables, objectives=objectives)

    assert sparow_soln_ID is not None  # ensure add method is returning soln ID 

    # ensure solution associated w/ sparow_soln_ID has expected structure/keys/values:
    soln_dict = solutions[sparow_soln_ID].to_dict()
    assert soln_dict == {
        "id": 0,
        "objectives": [
            {
                "index": None,
                "name": None,
                "suffix": {},
                "value": 76.5
            }
        ],
        "suffix": {},
        "variables": [
            {
                "discrete": False,
                "fixed": False,
                "index": None,
                "name": "x",
                "suffix": {},
                "value": 60.0
            },
            {
                "discrete": False,
                "fixed": False,
                "index": None,
                "name": "s[None,1].x",
                "suffix": {},
                "value": 15.0
            },
            {
                "discrete": False,
                "fixed": False,
                "index": None,
                "name": "s[None,2].x",
                "suffix": {},
                "value": 60.0
            }
        ]
    }

    assert solutions.get_pool_dicts() == {  # ensure sparow solution and pool manager have expected structure/keys/values
        "pool_1": {
            "metadata": {
                "as_solution_source": "sparow.solnpool.solnpool._sparow_as_solution",
                "context_name": "pool_1",
                "policy": "keep_all",
            },
            "pool_config": {},
            "solutions": {0: soln_dict},
        }
    }
