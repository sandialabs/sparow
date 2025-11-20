import random
import pytest
from forestlib.sp.scentobund import (
    single_scenario,
    single_bundle,
    sf_random,
    mf_paired,
    mf_random_nested,
    mf_random,
    similar_partitions,
    dissimilar_partitions,
    mf_kmeans_similar,
    mf_kmeans_dissimilar,
    kmeans_similar,
    kmeans_dissimilar,
    check_data_dict_keys,
    mf_bundle_from_list,
    bundle_from_list,
)


@pytest.fixture
def MF_data():
    return {
        "HF": {
            "scen_1": {"Demand": 3, "Probability": 0.4},
            "scen_0": {"Demand": 1, "Probability": 0.6},
        },
        "LF": {
            "scen_3": {"Demand": 4, "Probability": 0.2},
            "scen_2": {"Demand": 2, "Probability": 0.8},
        },
    }


@pytest.fixture
def MFpaired_data():
    return {
        "HF": {
            "scen_0": {"Demand": 3, "Probability": 0.2},
            "scen_1": {"Demand": 1, "Probability": 0.2},
            "scen_2": {"Demand": 2, "Probability": 0.3},
            "scen_3": {"Demand": 4, "Probability": 0.3},
        },
        "LF": {
            "scen_0": {"Demand": 4, "Probability": 0.2},
            "scen_1": {"Demand": 2, "Probability": 0.3},
            "scen_2": {"Demand": 1, "Probability": 0.3},
            "scen_3": {"Demand": 3, "Probability": 0.2},
        },
    }


@pytest.fixture
def SF_data():
    return {
        "HF": {
            "scen_1": {"Demand": 3, "Probability": 0.2},
            "scen_0": {"Demand": 1, "Probability": 0.3},
        },
        "LF": {
            "scen_3": {"Demand": 4, "Probability": 0.1},
            "scen_2": {"Demand": 2, "Probability": 0.4},
        },
    }


@pytest.fixture
def SF_missing_all_prob_data():
    return {
        "HF": {
            "scen_1": {"Demand": 3},
            "scen_0": {"Demand": 1},
        },
        "LF": {
            "scen_3": {"Demand": 4},
            "scen_2": {"Demand": 2},
        },
    }


@pytest.fixture
def SF_missing_some_prob_data():
    return {
        "HF": {
            "scen_1": {"Demand": 3, "Probability": 0.2},
            "scen_0": {"Demand": 1},
        },
        "LF": {
            "scen_3": {"Demand": 4},
            "scen_2": {"Demand": 2},
        },
    }


@pytest.fixture
def rand_data():
    return {
        "HF": {"scen_0": {"Demand": 1, "Probability": 0.6}},
        "LF": {"scen_2": {"Demand": 2, "Probability": 0.4}},
    }


@pytest.fixture
def rand_data_MF():
    return {
        "HF": {"scen_0": {"Demand": 1, "Probability": 1.0}},
        "LF": {"scen_2": {"Demand": 2, "Probability": 1.0}},
    }


@pytest.fixture
def imbalanced_data():
    return {
        "HF": {
            "rand_1": {"Demand": 3, "Probability": 0.5},
            "rand_0": {"Demand": 1, "Probability": 0.3},
        },
        "LF": {"rand_2": {"Demand": 4, "Probability": 0.2}},
    }


@pytest.fixture
def weird_key_names():
    return {
        "HF": {
            "s_1": {"weird_key_d": [3, 3], "weird_key_p": 0.5},
            "s_0": {"weird_key_d": [1, 1], "weird_key_p": 0.5},
        },
        "LF": {
            "s_2": {"weird_key_d": [4, 4], "weird_key_p": 0.2},
            "s_3": {"weird_key_d": [1.5, 1.5], "weird_key_p": 0.6},
            "s_4": {"weird_key_d": [5, 5], "weird_key_p": 0.2},
        },
    }

## TODO: update tests with this dict -R
@pytest.fixture
def sf_scenarios():
    return {
        "only_model_fidelity": {
            "s_1": {"demand": [3, 3], "probability": 0.25},
            "s_0": {"demand": [1, 1], "probability": 0.25},
            "s_2": {"demand": [4, 4], "probability": 0.1},
            "s_3": {"demand": [1.5, 1.5], "probability": 0.3},
            "s_4": {"demand": [5, 5], "probability": 0.1},
        },
    }


@pytest.fixture
def probable_key_names():
    return {
        "HF": {"s_1": {"d": [3, 3], "Pr": 0.5}, "s_0": {"d": [1, 1], "Pr": 0.5}},
        "LF": {
            "s_2": {"d": [4, 4], "Pr": 0.2},
            "s_3": {"d": [1.5, 1.5], "Pr": 0.6},
            "s_4": {"d": [5, 5], "Pr": 0.2},
        },
    }


@pytest.fixture
def similar_scenarios():
    return {
        "HF": {
            "scen_0": {"Demand": 4, "Probability": 0.2},
            "scen_1": {"Demand": 1, "Probability": 0.2},
            "scen_2": {"Demand": 2, "Probability": 0.3},
            "scen_3": {"Demand": 5, "Probability": 0.3},
        },
        "LF": {
            "scen_4": {"Demand": 5, "Probability": 0.2},
            "scen_5": {"Demand": 2, "Probability": 0.3},
            "scen_6": {"Demand": 1, "Probability": 0.3},
            "scen_7": {"Demand": 4, "Probability": 0.2},
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
        LFdemands = list(
            data[model][ls]["Demand"]
            for ls in LFscenarios[model]
            for model in models[1:]
        )

        # map each LF scenario to closest HF scenario using 1-norm of demand difference
        demand_diffs = {}
        for i in range(len(HFdemands)):
            for j in range(len(LFdemands)):
                demand_diffs[(i, j)] = abs(HFdemands[i] - LFdemands[j])

        return demand_diffs

    def test_check_data_dict_keys(
        self, weird_key_names, probable_key_names, SF_missing_all_prob_data
    ):
        assert single_bundle(probable_key_names) == {
            "bundle": {
                "scenarios": {
                    ("HF", "s_1"): 0.25,
                    ("HF", "s_0"): 0.25,
                    ("LF", "s_2"): 0.1,
                    ("LF", "s_3"): 0.3,
                    ("LF", "s_4"): 0.1,
                },
                "Probability": 1.0,
            },
        }

        assert single_bundle(
            weird_key_names,
            bundle_args={"demand_key": "weird_key_d", "probability_key": "weird_key_p"},
        ) == {
            "bundle": {
                "scenarios": {
                    ("HF", "s_1"): 0.25,
                    ("HF", "s_0"): 0.25,
                    ("LF", "s_2"): 0.1,
                    ("LF", "s_3"): 0.3,
                    ("LF", "s_4"): 0.1,
                },
                "Probability": 1.0,
            },
        }

        with pytest.raises(RuntimeError) as excinfo:
            single_bundle(weird_key_names)
        assert excinfo.type is RuntimeError

        with pytest.warns(
            UserWarning,
            match="No scenario probabilities are given; assuming uniform distribution.",
        ) as warninfo:
            single_bundle(SF_missing_all_prob_data)
        assert (
            warninfo[0].message.args[0]
            == "No scenario probabilities are given; assuming uniform distribution."
        )
        assert warninfo[0].category == UserWarning

        assert single_bundle(SF_missing_all_prob_data) == {
            "bundle": {
                "scenarios": {
                    ("HF", "scen_1"): 0.25,
                    ("HF", "scen_0"): 0.25,
                    ("LF", "scen_3"): 0.25,
                    ("LF", "scen_2"): 0.25,
                },
                "Probability": 1.0,
            }
        }

    def test_sf_model_weight_warnings(self, rand_data):
        with pytest.warns(
            UserWarning, match="Single fidelity schemes do not utilize model_weight"
        ) as warninfo:
            kmeans_similar(rand_data, model_weight={"HF": 2, "LF": 1})
        assert (
            warninfo[0].message.args[0]
            == "Single fidelity schemes do not utilize model_weight"
        )
        assert warninfo[0].category == UserWarning

    def test_mf_bundle_from_list(self, probable_key_names):
        with pytest.raises(RuntimeError) as excinfo:
            mf_bundle_from_list(probable_key_names)
        assert excinfo.type is RuntimeError

        assert mf_bundle_from_list(
            probable_key_names,
            bundle_args={
                "bundles": [
                    [("HF", "s_1"), ("LF", "s_2"), ("LF", "s_3")],
                    [("HF", "s_0"), ("LF", "s_4")],
                ],
            },
        ) == {
            "bundle_0": {
                "scenarios": {
                    ("HF", "s_1"): 0.3846153846153846,
                    ("LF", "s_2"): 0.15384615384615385,
                    ("LF", "s_3"): 0.46153846153846145,
                },
                "Probability": 0.65,
            },
            "bundle_1": {
                "scenarios": {
                    ("HF", "s_0"): 0.7142857142857143,
                    ("LF", "s_4"): 0.28571428571428575,
                },
                "Probability": 0.35,
            },
        }

    def test_bundle_from_list(self, sf_scenarios):
        with pytest.raises(RuntimeError) as excinfo:
            bundle_from_list(sf_scenarios)
        assert excinfo.type is RuntimeError

        assert bundle_from_list(
            sf_scenarios,
            bundle_args={
                "bundles": [
                    [("only_model_fidelity", "s_1"), ("only_model_fidelity", "s_2"), ("only_model_fidelity", "s_3")],
                    [("only_model_fidelity", "s_0"), ("only_model_fidelity", "s_4")],
                ],
            },
        ) == {
            "bundle_0": {
                "scenarios": {
                    ("only_model_fidelity", "s_1"): 0.25,
                    ("only_model_fidelity", "s_2"): 0.1,
                    ("only_model_fidelity", "s_3"): 0.3,
                },
                "Probability": 0.65,
            },
            "bundle_1": {
                "scenarios": {
                    ("only_model_fidelity", "s_0"): 0.25,
                    ("only_model_fidelity", "s_4"): 0.1,
                },
                "Probability": 0.35,
            },
        }

        assert bundle_from_list(
            sf_scenarios,
            bundle_args={
                "bundles": [["s_1", "s_2", "s_3"], ["s_0", "s_4"]],
            },
        ) == {
            "bundle_0": {
                "scenarios": {
                    ("only_model_fidelity", "s_1"): 0.25,
                    ("only_model_fidelity", "s_2"): 0.1,
                    ("only_model_fidelity", "s_3"): 0.3,
                },
                "Probability": 0.65,
            },
            "bundle_1": {
                "scenarios": {
                    ("only_model_fidelity", "s_0"): 0.25,
                    ("only_model_fidelity", "s_4"): 0.1,
                },
                "Probability": 0.35,
            },
        }

    def test_mf_kmeans_similar(self, similar_scenarios):
        assert mf_kmeans_similar(similar_scenarios) == {
            "bundle_4.0": {
                "scenarios": {("HF", "scen_0"): 0.5, ("LF", "scen_7"): 0.5},
                "Probability": 0.2,
            },
            "bundle_1.0": {
                "scenarios": {("HF", "scen_1"): 0.4, ("LF", "scen_6"): 0.6},
                "Probability": 0.25,
            },
            "bundle_2.0": {
                "scenarios": {("HF", "scen_2"): 0.5, ("LF", "scen_5"): 0.5},
                "Probability": 0.3,
            },
            "bundle_5.0": {
                "scenarios": {("HF", "scen_3"): 0.6, ("LF", "scen_4"): 0.4},
                "Probability": 0.25,
            },
        }
        assert mf_kmeans_similar(
            similar_scenarios, model_weight={"HF": 2, "LF": 1}
        ) == {
            "bundle_4.0": {
                "scenarios": {
                    ("HF", "scen_0"): 0.6666666666666666,
                    ("LF", "scen_7"): 0.3333333333333333,
                },
                "Probability": 0.2,
            },
            "bundle_1.0": {
                "scenarios": {
                    ("HF", "scen_1"): 0.5714285714285714,
                    ("LF", "scen_6"): 0.42857142857142855,
                },
                "Probability": 0.2333333333333333333,
            },
            "bundle_2.0": {
                "scenarios": {
                    ("HF", "scen_2"): 0.6666666666666666,
                    ("LF", "scen_5"): 0.3333333333333333,
                },
                "Probability": 0.3,
            },
            "bundle_5.0": {
                "scenarios": {
                    ("HF", "scen_3"): 0.7499999999999999,
                    ("LF", "scen_4"): 0.25,
                },
                "Probability": 0.26666666666666666,
            },
        }

    def test_mf_kmeans_dissimilar(self, similar_scenarios):
        assert mf_kmeans_dissimilar(similar_scenarios) == {
            "bundle_4.0": {"scenarios": {("HF", "scen_0"): 1.0}, "Probability": 0.1},
            "bundle_1.0": {
                "scenarios": {
                    ("HF", "scen_1"): 0.3333333333333333,
                    ("LF", "scen_7"): 0.3333333333333333,
                    ("LF", "scen_4"): 0.3333333333333333,
                },
                "Probability": 0.30000000000000004,
            },
            "bundle_2.0": {"scenarios": {("HF", "scen_2"): 1.0}, "Probability": 0.15},
            "bundle_5.0": {
                "scenarios": {
                    ("HF", "scen_3"): 0.3333333333333333,
                    ("LF", "scen_6"): 0.3333333333333333,
                    ("LF", "scen_5"): 0.3333333333333333,
                },
                "Probability": 0.44999999999999996,
            },
        }

        assert mf_kmeans_dissimilar(
            similar_scenarios, model_weight={"HF": 2, "LF": 1}
        ) == {
            "bundle_4.0": {"scenarios": {("HF", "scen_0"): 1.0}, "Probability": 0.2},
            "bundle_1.0": {
                "scenarios": {
                    ("HF", "scen_1"): 0.5,
                    ("LF", "scen_7"): 0.25,
                    ("LF", "scen_4"): 0.25,
                },
                "Probability": 0.2,
            },
            "bundle_2.0": {"scenarios": {("HF", "scen_2"): 1.0}, "Probability": 0.3},
            "bundle_5.0": {
                "scenarios": {
                    ("HF", "scen_3"): 0.5,
                    ("LF", "scen_6"): 0.25,
                    ("LF", "scen_5"): 0.25,
                },
                "Probability": 0.3,
            },
        }

    def test_similar_partitions(self, MF_data):
        assert similar_partitions(
            MF_data,
            models=["HF", "LF"],
            bundle_args={"distance_function": self.dist_map},
        ) == {
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
                "Probability": 0.5,
            },
        }

        assert similar_partitions(
            MF_data,
            model_weight={"HF": 3, "LF": 1},
            models=["HF", "LF"],
            bundle_args={"distance_function": self.dist_map},
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
                "Probability": 0.5,
            },
        }

    def test_dissimilar_partitions(self, similar_scenarios):
        assert dissimilar_partitions(
            similar_scenarios,
            models=["HF", "LF"],
            bundle_args={"distance_function": self.dist_map},
        ) == {
            "HF_scen_0": {
                "scenarios": {("HF", "scen_0"): 0.4, ("LF", "scen_6"): 0.6},
                "Probability": 0.25,
            },
            "HF_scen_1": {
                "scenarios": {("HF", "scen_1"): 0.5, ("LF", "scen_4"): 0.5},
                "Probability": 0.25,
            },
            "HF_scen_2": {
                "scenarios": {("HF", "scen_2"): 0.6, ("LF", "scen_4"): 0.4},
                "Probability": 0.25,
            },
            "HF_scen_3": {
                "scenarios": {("HF", "scen_3"): 0.5, ("LF", "scen_6"): 0.5},
                "Probability": 0.25,
            },
        }

        assert dissimilar_partitions(
            similar_scenarios,
            model_weight={"HF": 3, "LF": 1},
            models=["HF", "LF"],
            bundle_args={"distance_function": self.dist_map},
        ) == {
            "HF_scen_0": {
                "scenarios": {
                    ("HF", "scen_0"): 0.6666666666666666,
                    ("LF", "scen_6"): 0.33333333333333326,
                },
                "Probability": 0.25,
            },
            "HF_scen_1": {
                "scenarios": {
                    ("HF", "scen_1"): 0.7500000000000001,
                    ("LF", "scen_4"): 0.25,
                },
                "Probability": 0.25,
            },
            "HF_scen_2": {
                "scenarios": {
                    ("HF", "scen_2"): 0.8181818181818181,
                    ("LF", "scen_4"): 0.18181818181818185,
                },
                "Probability": 0.25,
            },
            "HF_scen_3": {
                "scenarios": {("HF", "scen_3"): 0.75, ("LF", "scen_6"): 0.25},
                "Probability": 0.25,
            },
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
        assert single_scenario(SF_data, models=["LF", "HF"]) == {
            "HF_scen_1": {"scenarios": {("HF", "scen_1"): 1.0}, "Probability": 0.2},
            "HF_scen_0": {"scenarios": {("HF", "scen_0"): 1.0}, "Probability": 0.3},
            "LF_scen_3": {"scenarios": {("LF", "scen_3"): 1.0}, "Probability": 0.1},
            "LF_scen_2": {"scenarios": {("LF", "scen_2"): 1.0}, "Probability": 0.4},
        }

        # checking logic with "fidelity" in bundle_args
        assert single_scenario(MF_data, models=["HF"]) == {
            "HF_scen_1": {"scenarios": {("HF", "scen_1"): 1.0}, "Probability": 0.4},
            "HF_scen_0": {"scenarios": {("HF", "scen_0"): 1.0}, "Probability": 0.6},
        }

        # checking logic with bundle_args that aren't "fidelity"
        assert single_scenario(
            SF_data,
            bundle_args={"some_other_arg": "arg"},
        ) == single_scenario(SF_data, bundle_args=None)

    def test_single_bundle(self, SF_data, MF_data):
        # check logic with no bundle args
        assert single_bundle(SF_data, models=["LF", "HF"]) == {
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
        assert single_bundle(MF_data, models=["LF"]) == {
            "bundle": {
                "scenarios": {("LF", "scen_3"): 0.2, ("LF", "scen_2"): 0.8},
                "Probability": 1.0,
            }
        }

        # check logic with bundle args that aren't fidelity
        assert single_bundle(
            SF_data,
            bundle_args={"some_other_arg": "arg"},
        ) == single_bundle(SF_data, bundle_args=None)

    def test_kmeans_similar(self, SF_data):
        # check logic with no bundle args
        assert kmeans_similar(SF_data) == {
            "bundle_1": {
                "scenarios": {
                    ("HF", "scen_0"): 0.42857142857142855,
                    ("LF", "scen_2"): 0.5714285714285715,
                },
                "Probability": 0.7,
            },
            "bundle_0": {
                "scenarios": {
                    ("HF", "scen_1"): 0.6666666666666666,
                    ("LF", "scen_3"): 0.3333333333333333,
                },
                "Probability": 0.30000000000000004,
            },
        }

        # check logic with bun_size
        assert kmeans_similar(SF_data, bundle_args={"bun_size": 1}) == {
            "bundle_3": {
                "scenarios": {("HF", "scen_0"): 1.0},
                "Probability": 0.3,
            },
            "bundle_2": {
                "scenarios": {("HF", "scen_1"): 1.0},
                "Probability": 0.2,
            },
            "bundle_1": {
                "scenarios": {("LF", "scen_2"): 1.0},
                "Probability": 0.4,
            },
            "bundle_0": {
                "scenarios": {("LF", "scen_3"): 1.0},
                "Probability": 0.1,
            },
        }
        assert kmeans_similar(SF_data, bundle_args={"bun_size": 4}) == {
            "bundle_0": {
                "scenarios": {
                    ("HF", "scen_0"): 0.3,
                    ("HF", "scen_1"): 0.2,
                    ("LF", "scen_2"): 0.4,
                    ("LF", "scen_3"): 0.1,
                },
                "Probability": 1.0,
            },
        }

        # ensure bun_size > num of scenarios returns error
        with pytest.raises(ValueError) as excinfo:
            kmeans_similar(SF_data, bundle_args={"bun_size": 5})
        assert excinfo.type is ValueError

    def test_kmeans_dissimilar(self, SF_data):
        assert kmeans_dissimilar(SF_data) == {
            "bundle_0": {
                "scenarios": {
                    ("HF", "scen_0"): 0.42857142857142855,
                    ("LF", "scen_2"): 0.5714285714285715,
                },
                "Probability": 0.7,
            },
            "bundle_1": {
                "scenarios": {
                    ("HF", "scen_1"): 0.6666666666666666,
                    ("LF", "scen_3"): 0.3333333333333333,
                },
                "Probability": 0.30000000000000004,
            },
        }

        # check logic with bun_size
        assert kmeans_dissimilar(SF_data, bundle_args={"bun_size": 1}) == {
            "bundle_0": {
                "scenarios": {
                    ("HF", "scen_0"): 0.42857142857142855,
                    ("LF", "scen_2"): 0.5714285714285715,
                },
                "Probability": 0.7,
            },
            "bundle_3": {
                "scenarios": {
                    ("HF", "scen_1"): 0.6666666666666666,
                    ("LF", "scen_3"): 0.3333333333333333,
                },
                "Probability": 0.30000000000000004,
            },
        }
        assert kmeans_dissimilar(SF_data, bundle_args={"bun_size": 4}) == {
            "bundle_0": {
                "scenarios": {
                    ("HF", "scen_0"): 0.3,
                    ("HF", "scen_1"): 0.2,
                    ("LF", "scen_2"): 0.4,
                    ("LF", "scen_3"): 0.1,
                },
                "Probability": 1.0,
            },
        }

        # ensure bun_size > num of scenarios returns error
        with pytest.raises(ValueError) as excinfo:
            kmeans_dissimilar(SF_data, bundle_args={"bun_size": 5})
        assert excinfo.type is ValueError
