import random

"""
bundle is a dictionary of dictionaries
    - keys are names of bundles
    - for each dictionary in bundle, keys are 'IDs' (i.e., which scenarios are in the bundle) and 'Probability'

specify which bundling scheme (function) is used via "bundle_scheme" in sp.py
"""


def bundle_by_fidelity(data, bundle_args=None):
    """Scenarios are bundled according to their fidelities"""

    if bundle_args is not None:
        raise RuntimeError(f"bundle_args not accepted by bundle_by_fidelity scheme")

    if any(
        data["scenarios"][i].get("Fidelity") == None
        for i in range(len(data["scenarios"]))
    ):
        raise RuntimeError(f"Fidelities not specified for all scenarios")

    bundle = {}
    bundle_names = list(set(map(lambda d: d["Fidelity"], data["scenarios"])))

    for fid in bundle_names:
        bundle[fid] = {}

    temp_dict = {}
    for fid in bundle_names:
        temp_dict[f"{fid}"] = []

    for i in range(len(data["scenarios"])):
        for fid in bundle_names:
            if data["scenarios"][i]["Fidelity"] == fid:
                temp_dict[f"{fid}"].append(data["scenarios"][i])

    for fid in bundle_names:  # each bundle assumed to have same probability
        bun_prob = sum(
            temp_dict[fid][i]["Probability"] for i in range(len(temp_dict[fid]))
        )
        bundle[f"{fid}"] = {
            "IDs": [
                temp_dict[f"{fid}"][i]["ID"] for i in range(len(temp_dict[f"{fid}"]))
            ],
            "Probability": bun_prob / len(bundle_names),
            "Scenario_Probabilities": {
                temp_dict[f"{fid}"][i]["ID"]: temp_dict[f"{fid}"][i]["Probability"]
                / bun_prob
                for i in range(len(temp_dict[f"{fid}"]))
            },
        }

    return bundle


def single_scenario(data, bundle_args):
    """Each scenario is its own bundle (i.e., no bundling)"""
    bundle = {}
    scens = []

    if bundle_args != None and "fidelity" in bundle_args:
        for i in range(len(data["scenarios"])):
            if data["scenarios"][i]["Fidelity"] == f"{bundle_args['fidelity']}":
                scens.append(data["scenarios"][i])
    else:
        assert bundle_args == None or "fidelity" not in bundle_args:
        for i in range(len(data["scenarios"])):
            scens.append(data["scenarios"][i])

    scen_prob = {
        j: scens[j].get("Probability", 1.0 / len(scens)) for j in range(len(scens))
    }
    for j in range(len(scens)):
        bun_prob = scen_prob[j]
        bundle[str(scens[j]["ID"])] = {
            "IDs": [scens[j]["ID"]],
            "Probability": bun_prob,
            "Scenario_Probabilities": {scens[j]["ID"]: scen_prob[j] / bun_prob},
        }

    return bundle


def single_bundle(data, bundle_args):
    """Every scenario in a single bundle (i.e., the subproblem is the master problem)"""
    bundle = {}

    if bundle_args != None and "fidelity" in bundle_args.keys():
        scens = []
        for i in range(len(data["scenarios"])):
            if data["scenarios"][i]["Fidelity"] == f"{bundle_args['fidelity']}":
                scens.append(data["scenarios"][i])
    else:
        assert bundle_args == None or "fidelity" not in bundle_args.keys()
        scens = []
        for i in range(len(data["scenarios"])):
            scens.append(data["scenarios"][i])

    bun_prob = sum(scens[j]["Probability"] for j in range(len(scens)))
    bundle["bundle"] = {
        "IDs": [scens[j]["ID"] for j in range(len(scens))],
        "Probability": bun_prob,
        "Scenario_Probabilities": {
            scens[j]["ID"]: scens[j]["Probability"] / bun_prob
            for j in range(len(scens))
        },
    }

    return bundle


def bundle_random_partition(data, bundle_args):
    """Each scenario is randomly assigned to a single bundle"""

    ## TODO: add seed to random number generator

    bundle = {}

    num_buns = bundle_args["num_buns"]

    if bundle_args != None and "fidelity" in bundle_args.keys():
        scens = []
        for i in range(len(data["scenarios"])):
            if data["scenarios"][i]["Fidelity"] == f"{bundle_args['fidelity']}":
                scens.append(data["scenarios"][i])
    else:
        assert bundle_args == None or "fidelity" not in bundle_args.keys()
        scens = []
        for i in range(len(data["scenarios"])):
            scens.append(data["scenarios"][i])

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
            "IDs": [
                scens[temp_bundle[bun_idx][l]]["ID"]
                for l in range(len(temp_bundle[bun_idx]))
            ],
            "Probability": bun_prob,
            "Scenario_Probabilities": {
                scens[temp_bundle[bun_idx][l]]["ID"]: scens[temp_bundle[bun_idx][l]][
                    "Probability"
                ]
                / bun_prob
                for l in range(len(temp_bundle[bun_idx]))
            },
        }

    return bundle


###################################################################################################################


scheme = {
    "bundle_by_fidelity": bundle_by_fidelity,
    "single_scenario": single_scenario,
    "single_bundle": single_bundle,
    "bundle_random_partition": bundle_random_partition,
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
        if abs(sum(bundle[key]["Scenario_Probabilities"].values()) - 1.0) > 1e-04:
            raise RuntimeError(
                f"Scenario probabilities within bundle {key} do not sum to 1"
            )

    return bundle


### TODO: add function that saves bundles to file
