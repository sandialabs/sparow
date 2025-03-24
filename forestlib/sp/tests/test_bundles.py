import random
import pytest
from forestlib.sp.scentobund import (
    single_scenario,
    single_bundle,
    bundle_random_partition,
    mf_paired,
    mf_random_nested,
    mf_random,
    similar_partitions,
)


@pytest.fixture
def MF_data():
    return {
        "HF": {
            "scen_1": {
                "Demand": 3,
                "Probability": 0.4,
            },
            "scen_0": {
                "Demand": 1,
                "Probability": 0.6,
            },
        },
        "LF": {
            "scen_3": {
                "Demand": 4,
                "Probability": 0.2,
            },
            "scen_2": {
                "Demand": 2,
                "Probability": 0.8,
            },
        },
    }


@pytest.fixture
def MFpaired_data():
    return {
        "HF": {
            "scen_0": {
                "Demand": 3,
                "Probability": 0.2,
            },
            "scen_1": {
                "Demand": 1,
                "Probability": 0.2,
            },
            "scen_2": {
                "Demand": 2,
                "Probability": 0.3,
            },
            "scen_3": {
                "Demand": 4,
                "Probability": 0.3,
            },
        },
        "LF": {
            "scen_0": {
                "Demand": 4,
                "Probability": 0.2,
            },
            "scen_1": {
                "Demand": 2,
                "Probability": 0.3,
            },
            "scen_2": {
                "Demand": 1,
                "Probability": 0.3,
            },
            "scen_3": {
                "Demand": 3,
                "Probability": 0.2,
            },
        },
    }


@pytest.fixture
def SF_data():
    return {
        "HF": {
            "scen_1": {
                "Demand": 3,
                "Probability": 0.2,
            },
            "scen_0": {
                "Demand": 1,
                "Probability": 0.3,
            },
        },
        "LF": {
            "scen_3": {
                "Demand": 4,
                "Probability": 0.1,
            },
            "scen_2": {
                "Demand": 2,
                "Probability": 0.4,
            },
        },
    }


@pytest.fixture
def rand_data():
    return {
        "HF": {
            "scen_0": {
                "Demand": 1,
                "Probability": 0.6,
            },
        },
        "LF": {
            "scen_2": {
                "Demand": 2,
                "Probability": 0.4,
            },
        },
    }


@pytest.fixture
def rand_data_MF():
    return {
        "HF": {
            "scen_0": {
                "Demand": 1,
                "Probability": 1.0,
            },
        },
        "LF": {
            "scen_2": {
                "Demand": 2,
                "Probability": 1.0,
            },
        },
    }


@pytest.fixture
def imbalanced_data():
    return {
        "HF": {
            "rand_1": {
                "Demand": 3,
                "Probability": 0.5,
            },
            "rand_0": {
                "Demand": 1,
                "Probability": 0.3,
            },
        },
        "LF": {
            "rand_2": {
                "Demand": 4,
                "Probability": 0.2,
            },
        },
    }


class TestBundleFunctions(object):

    def dist_map(self, data, models):
        model0 = models[0]
        
        HFscenarios = list(data[model0].keys())
        LFscenarios = {}  # all other models are LF
        for model in models[1:]:
            LFscenarios[model] = list(data[model].keys())

        HFdemands = list(data[model0][HFkey]["Demand"] for HFkey in HFscenarios)
        LFdemands = list(data[model][ls]["Demand"] for ls in LFscenarios[model] for model in models[1:])

        # map each LF scenario to closest HF scenario using 1-norm of demand difference
        demand_diffs = {}
        for i in range(len(HFdemands)):
            for j in range(len(LFdemands)):
                demand_diffs[(i,j)] = abs(HFdemands[i] - LFdemands[j])

        return demand_diffs

    def test_similar_partitions(self, MF_data):
        assert similar_partitions(MF_data, models=["HF", "LF"], bundle_args={"distance_function": self.dist_map}) == {
            "HF_scen_1": {
                "scenarios": {
                    ("HF", "scen_1"): 0.6666666666666666,
                    ("LF", "scen_3"): 0.3333333333333333,
                },
                "Probability": 0.5,
            },
            "HF_scen_0": {
                "scenarios": {
                    ("HF", "scen_0"): 0.42857142857142855,
                    ("LF", "scen_2"): 0.5714285714285715,
                    }, 
                    "Probability": 0.5},
        }

        assert similar_partitions(
            MF_data, model_weight={"HF": 3, "LF": 1}, models=["HF", "LF"], bundle_args={"distance_function": self.dist_map}
        ) == {
            "HF_scen_1": {
                "scenarios": {
                    ("HF", "scen_1"): 0.8571428571428572,
                    ("LF", "scen_3"): 0.14285714285714285,
                },
                "Probability": 0.5,
            },
            "HF_scen_0": {
                "scenarios": {
                    ("HF", "scen_0"): 0.6923076923076924, 
                    ("LF", "scen_2"): 0.30769230769230776,
                }, 
                "Probability": 0.5},
        }

    def test_mf_paired(self, MFpaired_data, imbalanced_data):
        assert mf_paired(
            MFpaired_data, model_weight={"HF": 1.0, "LF": 1.0}, models=["HF", "LF"]
        ) == {
            "scen_0": {
                "Probability": 0.25,
                "scenarios": {("HF", "scen_0"): 0.5, ("LF", "scen_0"): 0.5},
            },
            "scen_1": {
                "Probability": 0.25,
                "scenarios": {("HF", "scen_1"): 0.5, ("LF", "scen_1"): 0.5},
            },
            "scen_2": {
                "Probability": 0.25,
                "scenarios": {("HF", "scen_2"): 0.5, ("LF", "scen_2"): 0.5},
            },
            "scen_3": {
                "Probability": 0.25,
                "scenarios": {("HF", "scen_3"): 0.5, ("LF", "scen_3"): 0.5},
            },
        }

    def test_mf_random_nested(self, MFpaired_data, imbalanced_data):
        assert mf_random_nested(
            MFpaired_data,
            model_weight={"HF": 1.0, "LF": 1.0},
            models=["HF", "LF"],
            bundle_args=dict(seed=123456789),
        ) == {
            "HF_scen_0": {
                "Probability": 0.25,
                "scenarios": {("HF", "scen_0"): 0.5, ("LF", "scen_1"): 0.5},
            },
            "HF_scen_1": {
                "Probability": 0.25,
                "scenarios": {("HF", "scen_1"): 0.5, ("LF", "scen_2"): 0.5},
            },
            "HF_scen_2": {
                "Probability": 0.25,
                "scenarios": {("HF", "scen_2"): 0.5, ("LF", "scen_3"): 0.5},
            },
            "HF_scen_3": {
                "Probability": 0.25,
                "scenarios": {("HF", "scen_3"): 0.5, ("LF", "scen_0"): 0.5},
            },
        }
        assert mf_random_nested(
            MFpaired_data,
            model_weight={"HF": 3.0, "LF": 1.0},
            models=["HF", "LF"],
            bundle_args=dict(seed=123456789),
        ) == {
            "HF_scen_0": {
                "Probability": 0.25,
                "scenarios": {("HF", "scen_0"): 0.75, ("LF", "scen_1"): 0.25},
            },
            "HF_scen_1": {
                "Probability": 0.25,
                "scenarios": {("HF", "scen_1"): 0.75, ("LF", "scen_2"): 0.25},
            },
            "HF_scen_2": {
                "Probability": 0.25,
                "scenarios": {("HF", "scen_2"): 0.75, ("LF", "scen_3"): 0.25},
            },
            "HF_scen_3": {
                "Probability": 0.25,
                "scenarios": {("HF", "scen_3"): 0.75, ("LF", "scen_0"): 0.25},
            },
        }
        assert mf_random_nested(
            MFpaired_data,
            model_weight={"HF": 1.0, "LF": 1.0},
            models=["HF", "LF"],
            bundle_args=dict(LF=2, seed=1234567890),
        ) == {
            "HF_scen_0": {
                "Probability": 0.25,
                "scenarios": {
                    ("HF", "scen_0"): 0.3333333333333333,
                    ("LF", "scen_2"): 0.3333333333333333,
                    ("LF", "scen_3"): 0.3333333333333333,
                },
            },
            "HF_scen_1": {
                "Probability": 0.25,
                "scenarios": {
                    ("HF", "scen_1"): 0.3333333333333333,
                    ("LF", "scen_0"): 0.3333333333333333,
                    ("LF", "scen_2"): 0.3333333333333333,
                },
            },
            "HF_scen_2": {
                "Probability": 0.25,
                "scenarios": {
                    ("HF", "scen_2"): 0.3333333333333333,
                    ("LF", "scen_0"): 0.3333333333333333,
                    ("LF", "scen_3"): 0.3333333333333333,
                },
            },
            "HF_scen_3": {
                "Probability": 0.25,
                "scenarios": {
                    ("HF", "scen_3"): 0.3333333333333333,
                    ("LF", "scen_1"): 0.3333333333333333,
                    ("LF", "scen_2"): 0.3333333333333333,
                },
            },
        }

    def test_mf_random(self, MF_data, imbalanced_data):
        assert mf_random(
            MF_data,
            model_weight={"HF": 1.0, "LF": 1.0},
            models=["HF", "LF"],
            bundle_args=dict(seed=123456789),
        ) == {
            "HF_scen_0": {
                "Probability": 0.5,
                "scenarios": {("HF", "scen_0"): 0.5, ("LF", "scen_2"): 0.5},
            },
            "HF_scen_1": {
                "Probability": 0.5,
                "scenarios": {("HF", "scen_1"): 0.5, ("LF", "scen_2"): 0.5},
            },
        }

        assert mf_random(
            MF_data,
            model_weight={"HF": 3.0, "LF": 1.0},
            models=["HF", "LF"],
            bundle_args=dict(seed=123456789),
        ) == {
            "HF_scen_0": {
                "Probability": 0.5,
                "scenarios": {("HF", "scen_0"): 0.75, ("LF", "scen_2"): 0.25},
            },
            "HF_scen_1": {
                "Probability": 0.5,
                "scenarios": {("HF", "scen_1"): 0.75, ("LF", "scen_2"): 0.25},
            },
        }

        assert mf_random(
            MF_data,
            model_weight={"HF": 1.0, "LF": 1.0},
            models=["HF", "LF"],
            bundle_args=dict(LF=2, seed=1234567890),
        ) == {
            "HF_scen_0": {
                "Probability": 0.5,
                "scenarios": {
                    ("HF", "scen_0"): 0.3333333333333333,
                    ("LF", "scen_2"): 0.3333333333333333,
                    ("LF", "scen_3"): 0.3333333333333333,
                },
            },
            "HF_scen_1": {
                "Probability": 0.5,
                "scenarios": {
                    ("HF", "scen_1"): 0.3333333333333333,
                    ("LF", "scen_2"): 0.3333333333333333,
                    ("LF", "scen_3"): 0.3333333333333333,
                },
            },
        }

    def test_single_scenario(self, SF_data, MF_data):
        # checking logic with no bundle_args
        assert single_scenario(
            SF_data, model_weight={"HF": 1.0, "LF": 1.0}, models=["LF", "HF"]
        ) == {
            "HF_scen_1": {"scenarios": {("HF", "scen_1"): 1.0}, "Probability": 0.2},
            "HF_scen_0": {"scenarios": {("HF", "scen_0"): 1.0}, "Probability": 0.3},
            "LF_scen_3": {"scenarios": {("LF", "scen_3"): 1.0}, "Probability": 0.1},
            "LF_scen_2": {"scenarios": {("LF", "scen_2"): 1.0}, "Probability": 0.4},
        }

        # checking logic with "fidelity" in bundle_args
        assert single_scenario(
            MF_data, model_weight={"HF": 1.0, "LF": 1.0}, models=["HF"]
        ) == {
            "HF_scen_1": {"scenarios": {("HF", "scen_1"): 1.0}, "Probability": 0.4},
            "HF_scen_0": {"scenarios": {("HF", "scen_0"): 1.0}, "Probability": 0.6},
        }

        # checking logic with bundle_args that aren't "fidelity"
        assert single_scenario(
            SF_data,
            model_weight={"HF": 1.0, "LF": 1.0},
            bundle_args={"some_other_arg": "arg"},
        ) == single_scenario(
            SF_data, model_weight={"HF": 1.0, "LF": 1.0}, bundle_args=None
        )

    def test_single_bundle(self, SF_data, MF_data):
        # check logic with no bundle args
        assert single_bundle(
            SF_data, model_weight={"HF": 1.0, "LF": 1.0}, models=["LF", "HF"]
        ) == {
            "bundle": {
                "scenarios": {
                    ("HF", "scen_1"): 0.2,
                    ("HF", "scen_0"): 0.3,
                    ("LF", "scen_3"): 0.1,
                    ("LF", "scen_2"): 0.4,
                },
                "Probability": 1.0,
            }
        }

        # check logic with 'fidelity' in bundle args
        assert single_bundle(
            MF_data, model_weight={"HF": 1.0, "LF": 1.0}, models=["LF"]
        ) == {
            "bundle": {
                "scenarios": {("LF", "scen_3"): 0.2, ("LF", "scen_2"): 0.8},
                "Probability": 1.0,
            }
        }

        # check logic with bundle args that aren't fidelity
        assert single_bundle(
            SF_data,
            model_weight={"HF": 1.0, "LF": 1.0},
            bundle_args={"some_other_arg": "arg"},
        ) == single_bundle(
            SF_data, model_weight={"HF": 1.0, "LF": 1.0}, bundle_args=None
        )

    def Xtest_bundle_random_partition(self, rand_data, rand_data_MF):
        # check that no "num_buns" in bundle args returns error
        with pytest.raises(TypeError) as excinfo:
            bundle_random_partition(
                rand_data, model_weight={"HF": 1.0, "LF": 1.0}, bundle_args=None
            )
        assert excinfo.type is TypeError

        # check that "num_buns" larger than #scenarios returns error
        with pytest.raises(RuntimeError) as excinfo:
            bundle_random_partition(
                rand_data,
                model_weight={"HF": 1.0, "LF": 1.0},
                bundle_args={"num_buns": 4},
            )
        assert excinfo.type is RuntimeError

        # check that a non-integer/negative "num_buns" returns error
        with pytest.raises(ValueError) as excinfo:
            bundle_random_partition(rand_data, bundle_args={"num_buns": -1})
        assert excinfo.type is ValueError

        with pytest.raises(ValueError) as excinfo:
            bundle_random_partition(rand_data, bundle_args={"num_buns": 1.4})
        assert excinfo.type is ValueError

        # check logic with no 'fidelity' key in bundle_args
        rbundle0 = {
            "rand_0": {"scenarios": {"scen_0": 0.6}, "Probability": 1.0},
            "rand_1": {"scenarios": {"scen_2": 0.4}, "Probability": 1.0},
        }
        rbundle1 = {
            "rand_0": {"scenarios": {"scen_2": 0.4}, "Probability": 1.0},
            "rand_1": {"scenarios": {"scen_0": 0.6}, "Probability": 1.0},
        }
        assert (
            bundle_random_partition(
                rand_data,
                model_weight={"HF": 1.0, "LF": 1.0},
                bundle_args={"num_buns": 2},
            )
            == rbundle0
            or rbundle1
        )

        # check logic with 'fidelity' in bundle_args
        assert bundle_random_partition(
            rand_data_MF,
            model_weight={"HF": 1.0, "LF": 1.0},
            bundle_args={"num_buns": 1, "fidelity": "LF"},
        ) == {"rand_0": {"scenarios": {"scen_2": 1.0}, "Probability": 1.0}}

        # TODO: check logic with optional bundle args other than 'fidelity'
        assert bundle_random_partition(
            rand_data_MF,
            model_weight={"HF": 1.0, "LF": 1.0},
            bundle_args={"num_buns": 1, "some_other_arg": "arg"},
        ) == bundle_random_partition(rand_data_MF, bundle_args={"num_buns": 1})
