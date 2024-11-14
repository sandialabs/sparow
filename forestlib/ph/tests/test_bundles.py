from forestlib.ph.scentobund import bundle_by_fidelity
from forestlib.ph.scentobund import single_scenario
from forestlib.ph.scentobund import single_bundle
from forestlib.ph.scentobund import bundle_random_partition
import random
import pytest
import pdb


@pytest.fixture
def MF_data():
    return {
        "scenarios": [
            {
                "ID": "scen_1",
                "Fidelity": "HF",
                "Demand": 3,
                "Weight": 1,
                "Probability": 0.4,
            },
            {
                "ID": "scen_0",
                "Fidelity": "HF",
                "Demand": 1,
                "Weight": 1,
                "Probability": 0.6,
            },
            {
                "ID": "scen_3",
                "Fidelity": "LF",
                "Demand": 4,
                "Weight": 1,
                "Probability": 0.2,
            },
            {
                "ID": "scen_2",
                "Fidelity": "LF",
                "Demand": 2,
                "Weight": 1,
                "Probability": 0.8,
            },
        ]
    }


@pytest.fixture
def SF_data():
    return {
        "scenarios": [
            {
                "ID": "scen_1",
                "Fidelity": "HF",
                "Demand": 3,
                "Weight": 1,
                "Probability": 0.2,
            },
            {
                "ID": "scen_0",
                "Fidelity": "HF",
                "Demand": 1,
                "Weight": 1,
                "Probability": 0.3,
            },
            {
                "ID": "scen_3",
                "Fidelity": "LF",
                "Demand": 4,
                "Weight": 1,
                "Probability": 0.1,
            },
            {
                "ID": "scen_2",
                "Fidelity": "LF",
                "Demand": 2,
                "Weight": 1,
                "Probability": 0.4,
            },
        ]
    }


@pytest.fixture
def nofid_scen():
    return {"scenarios": [{"ID": "nofid", "Demand": 20, "Probability": 1.0}]}

@pytest.fixture
def rand_scens():
    return {
        "scenarios": [
            {
                "ID": "rand_1",
                "Fidelity": "HF",
                "Demand": 3,
                "Weight": 1,
                "Probability": 0.5,
            },
            {
                "ID": "rand_0",
                "Fidelity": "HF",
                "Demand": 1,
                "Weight": 1,
                "Probability": 0.3,
            },
            {
                "ID": "rand_2",
                "Fidelity": "LF",
                "Demand": 4,
                "Weight": 1,
                "Probability": 0.2,
            }
        ]
    }


class TestBundleFunctions(object):

    def test_bundle_by_fidelity(self, MF_data, nofid_scen):
        # check that nonempty bundle_args returns error
        with pytest.raises(RuntimeError) as excinfo:
            bundle_by_fidelity(MF_data, bundle_args={"test_arg": "test_arg"})
        assert excinfo.type is RuntimeError

        # check that scenarios are partitioned into bundles by their fidelities
        assert bundle_by_fidelity(MF_data) == {
            "HF": {"scenarios": {"scen_1": 0.4, "scen_0": 0.6}, "Probability": 0.5},
            "LF": {"scenarios": {"scen_3": 0.2, "scen_2": 0.8}, "Probability": 0.5}
        }

        # check that scenario with no fidelity key returns error
        with pytest.raises(RuntimeError) as excinfo:
            bundle_by_fidelity(nofid_scen)
        assert excinfo.type is RuntimeError

    def test_single_scenario(self, SF_data, MF_data):
        # checking logic with no bundle_args
        assert single_scenario(SF_data, bundle_args=None) == {
            "scen_1": {"scenarios": {"scen_1": 1.0}, "Probability": 0.2},
            "scen_0": {"scenarios": {"scen_0": 1.0}, "Probability": 0.3},
            "scen_3": {"scenarios": {"scen_3": 1.0}, "Probability": 0.1},
            "scen_2": {"scenarios": {"scen_2": 1.0}, "Probability": 0.4},
        }

        # checking logic w/ fidelity in bundle_args
        assert single_scenario(MF_data, bundle_args={"fidelity": "HF"}) == {
            "scen_1": {"scenarios": {"scen_1": 1.0}, "Probability": 0.4},
            "scen_0": {"scenarios": {"scen_0": 1.0}, "Probability": 0.6},
        }

        # checking logic with bundle_args that aren't fidelity
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

    def test_bundle_random_partition(self, rand_scens):
        # check that no "num_buns" in bundle args returns error
        with pytest.raises(TypeError) as excinfo:
            bundle_random_partition(rand_scens, bundle_args=None)
        assert excinfo.type is TypeError

        # check that "num_buns" larger than #scenarios returns error
        with pytest.raises(RuntimeError) as excinfo:
            bundle_random_partition(rand_scens, bundle_args={"num_buns": 4})
        assert excinfo.type is RuntimeError

        # check that a non-integer/negative "num_buns" returns error
        with pytest.raises(ValueError) as excinfo:
            bundle_random_partition(rand_scens, bundle_args={"num_buns": -1})
        assert excinfo.type is ValueError

        with pytest.raises(ValueError) as excinfo:
            bundle_random_partition(rand_scens, bundle_args={"num_buns": 2.4})
        assert excinfo.type is ValueError

        # TODO: check logic with no bundle args except seed
        assert bundle_random_partition(rand_scens, bundle_args={'num_buns': 3, 'seed': 1})

        # TODO: check logic with 'fidelity' in bundle_args

        # TODO: check logic with bundle args in addition to fidelity and seed

    def test_bundle_scheme(self):
        pass
