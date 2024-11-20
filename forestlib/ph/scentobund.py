import munch
import random

"""
bundle is a dictionary of dictionaries
    - keys are names of bundles
    - for each dictionary in bundle, keys are 'IDs' (i.e., which scenarios are in the bundle) and 'Probability'

specify which bundling scheme (function) is used via "bundle_scheme" in sp.py
"""


def scen_name(model, scenario):
    if model is None:
        return f"{scenario}"
    return f"{model}_{scenario}"


def scen_key(model, scenario):
    return (model, scenario)


def mf_paired(data, models=None, bundle_args=None):
    """
    Scenarios are paired according to their models

    Note that scenario probabilities specified for each model are ignored.
    """
    if models is None:
        models = list(data.keys())
    assert len(models) > 1, "Expecting multiple models for mf_paired"

    #
    # This bundling strategy requires that all models have the same scenarios.
    #
    model0 = models[0]
    scenarios = set(data[model0].keys())
    for model in models[1:]:
        assert scenarios == set(data[model].keys()), "All scenarios have the same keys"

    bundle = {}
    for s in scenarios:
        bundle[f"{s}"] = dict(
            scenarios={scen_key(model, s): 1.0 / len(models) for model in models},
            Probability=1.0 / len(scenarios),
        )

    return bundle


# ordered=False will randomize; default is True.
# assumes all scenario probs within each fidelity sum to 1!!!
def mf_ordered(data, models=None, bundle_args=None):
    """
    Scenarios are paired according to their models
    """
    if models is None:
        models = list(data.keys())
    assert len(models) > 1, "Expecting multiple models for mf_paired"

    if bundle_args is None:
        bundle_args = {"paired": True}

    # keys are model names, and values are how many scenarios for that model
    counts = {model: len(data[model]) for model in models}

    bundle = {}

    if bundle_args.get("ordered", False):
        #
        # scenarios from each model are paired by order they appear in scenario list
        #

        # max_fid is the first model in {models} with the largest number of scenarios.
        max_fid = max(range(len(models)), key=lambda i: (counts[models[i]], -i))

        for i in range(len(data[max_fid]["scenarios"])):
            bun_prob = (
                data[max_fid]["scenarios"][i]["Probability"]
                + data[min_fid]["scenarios"][i % counts[min_fid]]["Probability"]
            )
            bundle[f"ord_pair_{i}"] = {
                "scenarios": {
                    data[max_fid]["scenarios"][i]["ID"]: data[max_fid]["scenarios"][i][
                        "Probability"
                    ]
                    / bun_prob,
                    data[min_fid]["scenarios"][i % counts[min_fid]]["ID"]: data[
                        min_fid
                    ]["scenarios"][i % counts[min_fid]]["Probability"]
                    / bun_prob,
                },
                "Probability": bun_prob,
            }
        norm_factor = sum(bundle[key]["Probability"] for key in bundle.keys())
        for key in bundle.keys():
            bundle[key]["Probability"] /= norm_factor
    else:  # scenarios from each fid are randomly paired
        return RuntimeError(f"I haven't developed a non-ordered (random) method yet")

    # TODO: add random method

    return bundle


def single_scenario(data, models=None, bundle_args=None):
    """
    Each scenario is its own bundle (i.e., no bundling)
    """
    if models is None:
        models = list(data.keys())

    if all(
        "Probability" in sdata for model in models for sdata in data[model].values()
    ):
        #
        # Probability values have been specified for all scenarios, so we use the relative weight
        # of these probabilities
        #
        total_prob = sum(
            sdata["Probability"] for model in models for sdata in data[model].values()
        )
        bundle = {}
        for model in models:
            for s, sdata in data[model].items():
                bundle[scen_name(model, s)] = dict(
                    scenarios={scen_key(model, s): 1.0},
                    Probability=sdata["Probability"] / total_prob,
                )
    else:
        #
        # At least some of the scenarios are missing probability values, so we just assume
        # a uniform distribution.
        #
        total = sum(1 for model in models for sdata in data[model].values())
        bundle = {}
        for model in models:
            for s, sdata in data[model].items():
                bundle[scen_name(model, s)] = dict(
                    scenarios={scen_key(model, s): 1.0}, Probability=1.0 / total
                )

    return bundle


def single_bundle(data, models=None, bundle_args=None):
    """
    Combine scenarios from the specified models into a single bundle (i.e., the subproblem is the master problem).
    """
    if models is None:
        models = list(data.keys())

    if all(
        "Probability" in sdata for model in models for sdata in data[model].values()
    ):
        #
        # Probability values have been specified for all scenarios, so we use the relative weight
        # of these probabilities
        #
        bun_prob = 0
        for model in models:
            bun_prob += sum(sdata["Probability"] for sdata in data[model].values())

        scenarios = {}
        for model in models:
            for s, sdata in data[model].items():
                scenarios[scen_key(model, s)] = sdata["Probability"] / bun_prob
    else:
        #
        # At least some of the scenarios are missing probability values, so we just assume
        # a uniform distribution.
        #
        total = sum(1 for model in models for sdata in data[model].values())

        scenarios = {}
        for model in models:
            for s, sdata in data[model].items():
                scenarios[scen_key(model, s)] = 1.0 / total

    bundle = dict(bundle=dict(scenarios=scenarios, Probability=1.0))

    return bundle


def bundle_by_fidelity(data, models=None, bundle_args=None):
    """
    Scenarios are bundled according to their fidelities
    """
    if models is None:
        models = list(data.keys())

    bundle = {}
    for fid in models:
        bundle[fid] = single_bundle(data, models=[fid])["bundle"]
        # each bundle assumed to have same probability
        bundle[fid]["Probability"] = 1.0 / len(models)

    return bundle


def bundle_random_partition(data, models=None, bundle_args=None):
    """
    Each scenario is randomly assigned to a single bundle
    """
    if models is None:
        models = list(data.keys())

    # user can optionally set random seed
    seed_value = 972819128347298
    if bundle_args != None:
        seed_value = bundle_args.get("seed", seed_value)
    random.seed(seed_value)

    bundle = {}
    scens = []

    # solving model for one fidelity vs. multiple
    if bundle_args != None and "fidelity" in bundle_args.keys():
        for i in range(len(data[bundle_args["fidelity"]]["scenarios"])):
            scens.append(data[bundle_args["fidelity"]]["scenarios"][i])
    else:
        assert bundle_args == None or "fidelity" not in bundle_args.keys()
        for fid in data.keys():
            for i in range(len(data[fid]["scenarios"])):
                scens.append(data[fid]["scenarios"][i])

    # extracting number of bundles from bundle_args
    num_buns = bundle_args["num_buns"]

    if num_buns > len(scens):  # can't construct more bundles than num. of scenarios
        raise RuntimeError(f"Number of bundles must be <= number of scenarios")

    if num_buns < 0:  # number of bundles must be positive
        raise ValueError(f"Number of bundles must be positive")

    if (int(num_buns) == num_buns) == False:  # number of bundles must be integer
        raise ValueError(f"Number of bundles must be integer")

    base_bunsize = len(scens) // num_buns
    rem = len(scens) % num_buns

    bunsize = [None] * num_buns
    for i in range(len(bunsize)):
        bunsize[i] = base_bunsize

    for i in range(rem):
        bunsize[i] += 1

    for i in range(len(bunsize)):
        if any(bun == None for bun in bunsize):
            raise RuntimeError(f"No bundle size specified for bundle {i}")

    scen_idx = []  # temporary list
    for idx, scen in enumerate(
        scens
    ):  # using indices rather than IDs to allow for non-numeric scenario IDs
        scen_idx.append(idx)

    temp_bundle = []
    for bun_idx in range(num_buns):
        temp_list = random.sample(scen_idx, bunsize[bun_idx])
        temp_bundle.append(temp_list)
        for temp_scen_idx in temp_list:
            scen_idx.remove(temp_scen_idx)

    if len(scen_idx) != 0:
        raise RuntimeError(f"Scenarios {scen_idx} are not assigned to a bundle")

    for bun_idx in range(num_buns):
        bun_prob = sum(
            scens[temp_bundle[bun_idx][l]]["Probability"]
            for l in range(len(temp_bundle[bun_idx]))
        )
        bundle[f"rand_{bun_idx}"] = {
            "scenarios": {
                scens[temp_bundle[bun_idx][l]]["ID"]: scens[temp_bundle[bun_idx][l]][
                    "Probability"
                ]
                / bun_prob
                for l in range(len(temp_bundle[bun_idx]))
            },
            "Probability": bun_prob,
        }

    return bundle


###################################################################################################################

scheme = {
    "bundle_by_fidelity": bundle_by_fidelity,
    "single_scenario": single_scenario,
    "single_bundle": single_bundle,
    "bundle_random_partition": bundle_random_partition,
    "mf_paired": mf_paired,
    "mf_ordered": mf_ordered,
}


def bundle_scheme(data, scheme_str, models, bundle_args=None):
    bundle = scheme[scheme_str](data, models, bundle_args)

    # Return error if bundle probabilities do not sum to 1
    if abs(sum(b["Probability"] for b in bundle.values()) - 1.0) > 1e-04:
        raise RuntimeError(
            f"Bundle probabilities sum to {sum(bundle[key]['Probability'] for key in bundle)}"
        )

    # Return error if scenario probabilities within a bundle do not sum to 1
    for key in bundle:
        if abs(sum(bundle[key]["scenarios"].values()) - 1.0) > 1e-04:
            raise RuntimeError(
                f"Scenario probabilities within bundle {key} do not sum to 1"
            )

    return bundle


class BundleObj(object):

    def __init__(self, *, data, scheme, models, bundle_args):
        self.bundle_scheme_str = scheme
        self.bundle_models = models
        self.bundle_args = bundle_args
        bundles = bundle_scheme(data, scheme, models, bundle_args)
        self._bundles = {
            key: munch.Munch(
                probability=bundles[key]["Probability"],
                scenarios=list(sorted(bundles[key]["scenarios"].keys())),
                scenario_probability=bundles[key]["scenarios"],
            )
            for key in bundles
        }

    def __contains__(self, key):
        return key in self._bundles

    def __getitem__(self, key):
        assert (
            key in self._bundles
        ), f"Unexpected key {key} {type(key)}.  Valid keys: {list(self._bundles.keys())}"
        return self._bundles[key]

    def __iter__(self):
        for key in self._bundles:
            yield key

    def keys(self):
        for key in self._bundles:
            yield key
