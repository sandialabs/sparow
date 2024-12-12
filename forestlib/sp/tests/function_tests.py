import json
import random

with open("scenlist.json", "r") as file:
    data = json.load(file)


def mf_paired(
    data, bundle_args=None
):  # ordered=False will randomize; default is True. assumes all scenario probs within each fidelity sum to 1!!!
    """
    Scenarios are paired according to their fidelities
    this will only work with exactly two fidelities!!!!!!
    """
    counts = (
        {}
    )  # keys are fidelity, values are how many scenarios of that fidelity are in scenario list
    for fid in data.keys():
        counts[fid] = len(data[fid]["scenarios"])

    fidelities = list(data.keys())
    """
        max_fid is the fidelity with the largest number of scenarios; max_fid and min_fid are arbitrarily set to the first/second
        in the scenario list respectively if each fidelity has the same number of scenarios
    """
    if all(counts[fidelities[0]] == counts[fid] for fid in fidelities):
        max_fid = fidelities[0]
        min_fid = fidelities[1]
    else:
        max_fid = max(counts, key=counts.get)
        min_fid = min(counts, key=counts.get)

    bundle = {}

    if (
        bundle_args["ordered"] == True
    ):  # scenarios from each fid are paired by order they appear in scenario list
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


def bundle_by_fidelity(
    data, bundle_args=None
):  # assumes all scenario probs within each fidelity sum to 1!!!
    """Scenarios are bundled according to their fidelities"""

    bundle = {}
    bundle_names = list(data.keys())

    for fid in bundle_names:
        bundle[fid] = {}

    for fid in bundle_names:  # each bundle assumed to have same probability
        bun_prob = sum(
            data[fid]["scenarios"][i]["Probability"]
            for i in range(len(data[fid]["scenarios"]))
        )
        bundle[f"{fid}"] = {
            "scenarios": {
                data[f"{fid}"]["scenarios"][i]["ID"]: data[f"{fid}"]["scenarios"][i][
                    "Probability"
                ]
                / bun_prob
                for i in range(len(data[f"{fid}"]["scenarios"]))
            },
            "Probability": bun_prob / len(bundle_names),
        }

    return bundle


def single_scenario(
    data, bundle_args=None
):  # input fidelity as bundle arg if only solving one model fidelity
    """Each scenario is its own bundle (i.e., no bundling)"""
    bundle = {}
    scens = []

    if bundle_args != None and "fidelity" in bundle_args.keys():
        for i in range(len(data[bundle_args["fidelity"]]["scenarios"])):
            scens.append(data[bundle_args["fidelity"]]["scenarios"][i])
    else:
        assert bundle_args == None or "fidelity" not in bundle_args.keys()
        for fid in data.keys():
            for i in range(len(data[fid]["scenarios"])):
                scens.append(data[fid]["scenarios"][i])

    scen_prob = {
        j: scens[j].get("Probability", 1.0 / len(scens)) for j in range(len(scens))
    }
    for j in range(len(scens)):
        bundle[str(scens[j]["ID"])] = {
            "scenarios": {scens[j]["ID"]: 1.0},
            "Probability": scen_prob[j],
        }

    return bundle


def single_bundle(
    data, bundle_args=None
):  # input fidelity as bundle arg if only solving one model fidelity
    """Every scenario in a single bundle (i.e., the subproblem is the master problem)"""
    bundle = {}
    scens = []

    if bundle_args != None and "fidelity" in bundle_args.keys():
        for i in range(len(data[bundle_args["fidelity"]]["scenarios"])):
            scens.append(data[bundle_args["fidelity"]]["scenarios"][i])
    else:
        assert bundle_args == None or "fidelity" not in bundle_args.keys()
        for fid in data.keys():
            for i in range(len(data[fid]["scenarios"])):
                scens.append(data[fid]["scenarios"][i])

    bun_prob = sum(scens[j]["Probability"] for j in range(len(scens)))
    bundle["bundle"] = {
        "scenarios": {
            scens[j]["ID"]: scens[j]["Probability"] / bun_prob
            for j in range(len(scens))
        },
        "Probability": bun_prob,
    }

    return bundle


def bundle_random_partition(
    data, bundle_args
):  # input fidelity as bundle arg if only solving one model fidelity!!!
    """Each scenario is randomly assigned to a single bundle"""

    bundle = {}
    scens = []

    # user can optionally set random seed
    if bundle_args != None and "seed" in bundle_args.keys():
        seed_value = bundle_args["seed"]
    else:
        seed_value = 972819128347298
    random.seed(seed_value)

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


scheme = {
    "bundle_by_fidelity": bundle_by_fidelity,
    "single_scenario": single_scenario,
    "single_bundle": single_bundle,
    "bundle_random_partition": bundle_random_partition,
    "mf_paired": mf_paired,
}


def bundle_scheme(data, scheme_str, bundle_args=None):
    bundle = scheme[scheme_str](data, bundle_args)

    # Return error if bundle probabilities do not sum to 1
    if abs(sum(bundle[key]["Probability"] for key in bundle.keys()) - 1.0) > 1e-04:
        raise RuntimeError(
            f"Bundle probabilities sum to {sum(bundle[key]['Probability'] for key in bundle.keys())}"
        )

    # Return error if scenario probabilities within a bundle do not sum to 1
    for key in bundle.keys():
        if abs(sum(bundle[key]["scenarios"].values()) - 1.0) > 1e-04:
            raise RuntimeError(
                f"Scenario probabilities within bundle {key} do not sum to 1"
            )

    return bundle


bundle = bundle_scheme(
    data, "single_scenario", bundle_args={"fidelity": "HF", "num_buns": 3}
)
print(bundle)
