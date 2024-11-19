from forestlib.ph.scentobund import (
    bundle_by_fidelity,
    single_scenario,
    single_bundle,
    bundle_random_partition,
    mf_paired,
)
import random
import pytest


@pytest.fixture
def MF_data():
    return {
        "HF": {
            "scenarios": [
                {
                    "ID": "scen_1",
                    "Demand": 3,
                    "Probability": 0.4,
                },
                {
                    "ID": "scen_0",
                    "Demand": 1,
                    "Probability": 0.6,
                },
            ]
        },
        "LF": {
            "scenarios": [
                {
                    "ID": "scen_3",
                    "Demand": 4,
                    "Probability": 0.2,
                },
                {
                    "ID": "scen_2",
                    "Demand": 2,
                    "Probability": 0.8,
                },
            ]
        },
    }


@pytest.fixture
def SF_data():
    return {
        "HF": {
            "scenarios": [
                {
                    "ID": "scen_1",
                    "Demand": 3,
                    "Probability": 0.2,
                },
                {
                    "ID": "scen_0",
                    "Demand": 1,
                    "Probability": 0.3,
                },
            ]
        },
        "LF": {
            "scenarios": [
                {
                    "ID": "scen_3",
                    "Demand": 4,
                    "Probability": 0.1,
                },
                {
                    "ID": "scen_2",
                    "Demand": 2,
                    "Probability": 0.4,
                },
            ]
        },
    }


@pytest.fixture
def rand_data():
    return {
        "HF": {
            "scenarios": [
                {
                    "ID": "scen_0",
                    "Demand": 1,
                    "Probability": 0.6,
                },
            ]
        },
        "LF": {
            "scenarios": [
                {
                    "ID": "scen_2",
                    "Demand": 2,
                    "Probability": 0.4,
                },
            ]
        },
    }


@pytest.fixture
def rand_data_MF():
    return {
        "HF": {
            "scenarios": [
                {
                    "ID": "scen_0",
                    "Demand": 1,
                    "Probability": 1.0,
                },
            ]
        },
        "LF": {
            "scenarios": [
                {
                    "ID": "scen_2",
                    "Demand": 2,
                    "Probability": 1.0,
                },
            ]
        },
    }


@pytest.fixture
def imbalanced_data():
    return {
        "HF": {
            "scenarios": [
                {
                    "ID": "rand_1",
                    "Demand": 3,
                    "Probability": 0.5,
                },
                {
                    "ID": "rand_0",
                    "Demand": 1,
                    "Probability": 0.3,
                },
            ]
        },
        "LF": {
            "scenarios": [
                {
                    "ID": "rand_2",
                    "Demand": 4,
                    "Probability": 0.2,
                },
            ]
        },
    }


class TestBundleFunctions(object):

    def test_mf_paired(self, MF_data, imbalanced_data):
        # check bundling when #HF scenarios = #LF scenarios:
        assert mf_paired(MF_data, bundle_args={"ordered": True}) == {
            "ord_pair_0": {
                "scenarios": {
                    "scen_1": 0.6666666666666666,
                    "scen_3": 0.3333333333333333,
                },
                "Probability": 0.30000000000000004,
            },
            "ord_pair_1": {
                "scenarios": {
                    "scen_0": 0.4285714285714286,
                    "scen_2": 0.5714285714285715,
                },
                "Probability": 0.7,
            },
        }

        # check bundling when # HF scens /= # LF scens:
        assert mf_paired(imbalanced_data, bundle_args={"ordered": True}) == {
            "ord_pair_0": {
                "scenarios": {
                    "rand_1": 0.7142857142857143,
                    "rand_2": 0.28571428571428575,
                },
                "Probability": 0.5833333333333334,
            },
            "ord_pair_1": {
                "scenarios": {"rand_0": 0.6, "rand_2": 0.4},
                "Probability": 0.4166666666666667,
            },
        }

    def test_bundle_by_fidelity(self, MF_data):
        # check that nonempty bundle_args returns the same bundle as empty bundle_args
        assert bundle_by_fidelity(
            MF_data, bundle_args={"test_arg": "test_arg"}
        ) == bundle_by_fidelity(MF_data, bundle_args=None)

        # check that scenarios are partitioned into bundles by their fidelities
        assert bundle_by_fidelity(MF_data) == {
            "HF": {"scenarios": {"scen_1": 0.4, "scen_0": 0.6}, "Probability": 0.5},
            "LF": {"scenarios": {"scen_3": 0.2, "scen_2": 0.8}, "Probability": 0.5},
        }

    def test_single_scenario(self, SF_data, MF_data):
        # checking logic with no bundle_args
        assert single_scenario(SF_data, bundle_args=None) == {
            "scen_1": {"scenarios": {"scen_1": 1.0}, "Probability": 0.2},
            "scen_0": {"scenarios": {"scen_0": 1.0}, "Probability": 0.3},
            "scen_3": {"scenarios": {"scen_3": 1.0}, "Probability": 0.1},
            "scen_2": {"scenarios": {"scen_2": 1.0}, "Probability": 0.4},
        }

        # checking logic with "fidelity" in bundle_args
        assert single_scenario(MF_data, bundle_args={"fidelity": "HF"}) == {
            "scen_1": {"scenarios": {"scen_1": 1.0}, "Probability": 0.4},
            "scen_0": {"scenarios": {"scen_0": 1.0}, "Probability": 0.6},
        }

        # checking logic with bundle_args that aren't "fidelity"
        assert single_scenario(
            SF_data, bundle_args={"some_other_arg": "arg"}
        ) == single_scenario(SF_data, bundle_args=None)

    def test_single_bundle(self, SF_data, MF_data):
        # check logic with no bundle args
        assert single_bundle(SF_data) == {
            "bundle": {
                "scenarios": {
                    "scen_1": 0.2,
                    "scen_0": 0.3,
                    "scen_3": 0.1,
                    "scen_2": 0.4,
                },
                "Probability": 1.0,
            }
        }

        # check logic with 'fidelity' in bundle args
        assert single_bundle(MF_data, bundle_args={"fidelity": "LF"}) == {
            "bundle": {"scenarios": {"scen_3": 0.2, "scen_2": 0.8}, "Probability": 1.0}
        }

        # check logic with bundle args that aren't fidelity
        assert single_bundle(
            SF_data, bundle_args={"some_other_arg": "arg"}
        ) == single_bundle(SF_data, bundle_args=None)

    def test_bundle_random_partition(self, rand_data, rand_data_MF):
        # check that no "num_buns" in bundle args returns error
        with pytest.raises(TypeError) as excinfo:
            bundle_random_partition(rand_data, bundle_args=None)
        assert excinfo.type is TypeError

        # check that "num_buns" larger than #scenarios returns error
        with pytest.raises(RuntimeError) as excinfo:
            bundle_random_partition(rand_data, bundle_args={"num_buns": 4})
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
            bundle_random_partition(rand_data, bundle_args={"num_buns": 2}) == rbundle0
            or rbundle1
        )

        # check logic with 'fidelity' in bundle_args
        assert bundle_random_partition(
            rand_data_MF, bundle_args={"num_buns": 1, "fidelity": "LF"}
        ) == {"rand_0": {"scenarios": {"scen_2": 1.0}, "Probability": 1.0}}

        # TODO: check logic with optional bundle args other than 'fidelity'
        assert bundle_random_partition(
            rand_data_MF, bundle_args={"num_buns": 1, "some_other_arg": "arg"}
        ) == bundle_random_partition(rand_data_MF, bundle_args={"num_buns": 1})
