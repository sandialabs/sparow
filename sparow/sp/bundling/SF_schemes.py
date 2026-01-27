import random
from sklearn.cluster import KMeans
import numpy as np
import numbers
import warnings

import sparow.logs
import sparow.sp.bundling.bundling_helper_functions as bf

logger = sparow.logs.logger

"""
******************* SINGLE-FIDELITY SCHEMES *******************
"""

def single_scenario(data, models=None, bundle_args=None, *args):
    """
    Each scenario is its own bundle (i.e., no bundling)
    """
    if models is None:
        models = list(data.keys())

    # model0 = models[0]
    # pkey = bf.check_data_dict_keys(data, model0, bundle_args)[1]
    pkey = "Probability"

    if all(pkey in sdata for model in models for sdata in data[model].values()):
        #
        # Probability values have been specified for all scenarios, so we use the relative weight
        # of these probabilities
        #
        total_prob = sum(
            sdata[pkey] for model in models for sdata in data[model].values()
        )
        bundle = {}
        for model in models:
            for s, sdata in data[model].items():
                bundle[bf.scen_name(model, s)] = dict(
                    scenarios={bf.scen_key(model, s): 1.0},
                    Probability=sdata[pkey] / total_prob,
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
                bundle[bf.scen_name(model, s)] = dict(
                    scenarios={bf.scen_key(model, s): 1.0}, Probability=1.0 / total
                )

    return bundle


def single_bundle(data, models=None, bundle_args=None, *args):
    """
    Combine scenarios from the specified models into a single bundle (i.e., the subproblem is the master problem).
    """
    logger.info("Using single_bundle scheme (extensive form solve).")

    if models is None:
        models = list(data.keys())

    model0 = models[0]
    pkey = bf.check_data_dict_keys(data, model0, bundle_args)[1]

    if pkey == None:
        #
        # No scenario probabilities are given, so we just assume
        # a uniform distribution.
        #
        total = sum(1 for model in models for sdata in data[model].values())

        scenarios = {}
        for model in models:
            for s, sdata in data[model].items():
                scenarios[bf.scen_key(model, s)] = 1.0 / total
    else:
        #
        # Probability values have been specified for all scenarios, so we use the relative weight
        # of these probabilities
        #
        bun_prob = 0
        for model in models:
            bun_prob += sum(sdata[pkey] for sdata in data[model].values())

        scenarios = {}
        for model in models:
            for s, sdata in data[model].items():
                scenarios[bf.scen_key(model, s)] = sdata[pkey] / bun_prob

    bundle = dict(bundle=dict(scenarios=scenarios, Probability=1.0))

    return bundle


def bundle_from_list(data, models=None, bundle_args=None):
    """
    This scheme accepts a list of lists provided by the user, where each inner list is a bundle.
    Assumes user will pass in the list of lists to bundle_args with key "bundles",
        e.g., bundle_args["bundles"] = [[("HF", "scen_1"), ("HF", "scen_3")], [("HF", "scen_0"), ("HF", "scen_2")]], OR
              bunlde_args["bundles"] = [["scen_1", "scen_3"], ["scen_0", "scen_2"]] (if scenario names are unique).
    This scheme is well-suited for users who wish to specify their own bundles.
    """

    if bundle_args:
        list_of_bundles = bundle_args.get("bundles")
    else:
        list_of_bundles = None
    if list_of_bundles is None or len(list_of_bundles) == 0:
        raise RuntimeError(f"bundle_from_list scheme requires 'bundles' key")

    if models is None:
        models = list(data.keys())

    model0 = models[0]  # the first model in models is assumed to be the HF model
    pkey = bf.check_data_dict_keys(data, model0, bundle_args)[1]

    tuple_type = True
    string_type = True
    # checks if user specified tuples or scenario string names:
    for bundle_list in list_of_bundles:
        if all(isinstance(scen, tuple) for scen in bundle_list) and tuple_type:
            string_type = False
        elif all(isinstance(scen, str) for scen in bundle_list) and string_type:
            tuple_type = False
        else:
            raise RuntimeError(
                f"User-specified bundle list contains inconsistent entries."
            )

    bundle = {}
    if tuple_type:
        for bundle_list_idx, bundle_list in enumerate(list_of_bundles):
            bundle[f"bundle_{bundle_list_idx}"] = dict(
                scenarios={
                    bundle_list[b_tuple_idx]: data[f"{b_tuple[0]}"][f"{b_tuple[1]}"][
                        pkey
                    ]
                    for b_tuple_idx, b_tuple in enumerate(bundle_list)
                },
                Probability=sum(
                    data[f"{b_tuple[0]}"][f"{b_tuple[1]}"][pkey]
                    for b_tuple in bundle_list
                ),
            )
    else:
        assert (
            string_type == True
        ), "Inner bundle lists must contain only strings or only tuples"
        for bundle_list_idx, bundle_list in enumerate(list_of_bundles):
            bundle[f"bundle_{bundle_list_idx}"] = dict(
                scenarios={
                    (model0, bundle_list[b_idx]): data[model0][f"{b}"][pkey]
                    for b_idx, b in enumerate(bundle_list)
                },
                Probability=sum(data[model0][f"{b}"][pkey] for b in bundle_list),
            )

    return bundle


def kmeans_similar(data, models=None, bundle_args=None):
    """
    Each scenario is paired by closest distance
        - bun_size (approx. size of each bundle) can be passed into bundle_args. default is 2.
        - ensure there are no duplicate/redundant scenarios before using this!
        - all scenarios must have unique names!
    """

    if models is None:
        models = list(data.keys())

    model0 = models[0]  # the first model in models is assumed to be the HF model
    dkey = bf.check_data_dict_keys(data, model0, bundle_args, dkey_required=True)[0]
    pkey = bf.check_data_dict_keys(data, model0, bundle_args, dkey_required=True)[1]

    if bundle_args is not None:  # default bundle size is 2
        bun_size = bundle_args.get("bun_size", 2)
    else:
        bun_size = 2

    num_scens = sum(len(data[model]) for model in models)  # total number of scenarios
    if bun_size > num_scens:
        raise ValueError(f"Bundle size cannot exceed number of scenarios")

    num_centers = -(
        num_scens // -bun_size
    )  # number of bundle centers (NOT NECESSARILY the same as number of bundles!!)
    all_scens = {
        sname: sval for model in models for (sname, sval) in data[model].items()
    }

    if isinstance(data[model0][next(iter(all_scens))][dkey], numbers.Number) == True:
        arr = np.array([all_scens[s][dkey] for s in all_scens.keys()])
        X = arr.reshape(
            -1, 1
        )  # array of scenario demands needs to be reshaped if demand is a number
    else:
        X = np.array([all_scens[s][dkey] for s in all_scens.keys()])

    kmeans = KMeans(n_clusters=num_centers, random_state=0, n_init="auto").fit(
        X
    )  # find bundle centers
    s_assign = kmeans.labels_  # list of closest bundle centers
    sbmap = {
        s: int(s_assign[ind_s]) for ind_s, s in enumerate(iter(all_scens))
    }  # map scenarios to closest bundle center

    bundle = {}
    for model in models:
        for s in data[model]:
            if (
                f"bundle_{sbmap[s]}" in bundle
            ):  # ensures empty bundles aren't created if centers have no mapped scenarios
                bundle[f"bundle_{sbmap[s]}"]["scenarios"].update(
                    {bf.scen_key(model, s): data[model][s][pkey]}
                )
            else:
                bundle[f"bundle_{sbmap[s]}"] = dict(
                    scenarios={bf.scen_key(model, s): data[model][s][pkey]},
                    Probability=0,
                )
    # bundle probability is normalized sum of scenario probabilities within bundle
    for bkey in bundle.keys():
        bundle[bkey]["Probability"] = sum(bundle[bkey]["scenarios"].values())
    # normalize bundle probabilities
    bundle_prob_norm = sum(bundle[bkey]["Probability"] for bkey in bundle.keys())
    for bkey in bundle.keys():
        bundle[bkey]["Probability"] *= 1 / bundle_prob_norm

    # normalization term for if some centers have no mapped scenarios
    fewer_centers_norm = sum(bundle[bkey]["Probability"] for bkey in bundle.keys())
    for bkey in bundle.keys():
        bundle[bkey]["Probability"] *= 1 / fewer_centers_norm
        norm_factor = sum(
            bundle[bkey]["scenarios"].values()
        )  # normalizing bundle probabilities
        for skey in bundle[bkey]["scenarios"].keys():
            bundle[bkey]["scenarios"][skey] *= 1 / norm_factor

    return bundle


def kmeans_dissimilar(data, models=None, bundle_args=None):
    """
    Each scenario is paired by furthest distance
        - bun_size (approx. size of each bundle) can be passed into bundle_args. default is 2.
        - ensure there are no duplicate/redundant scenarios before using this!
        - all scenarios must have unique names!
    """

    if models is None:
        models = list(data.keys())

    model0 = models[0]  # the first model in models is assumed to be the HF model
    dkey = bf.check_data_dict_keys(data, model0, bundle_args, dkey_required=True)[0]
    pkey = bf.check_data_dict_keys(data, model0, bundle_args, dkey_required=True)[1]

    if bundle_args is not None:  # default bundle size is 2
        bun_size = bundle_args.get("bun_size", 2)
    else:
        bun_size = 2

    num_scens = sum(len(data[model]) for model in models)  # total number of scenarios
    if bun_size > num_scens:
        raise ValueError(f"Bundle size cannot exceed number of scenarios")

    num_centers = -(
        num_scens // -bun_size
    )  # number of bundle centers (NOT NECESSARILY the same as number of bundles!!)
    all_scens = {
        sname: sval for model in models for (sname, sval) in data[model].items()
    }

    if isinstance(data[model0][next(iter(all_scens))][dkey], numbers.Number) == True:
        arr = np.array([all_scens[s][dkey] for s in all_scens.keys()])
        X = arr.reshape(
            -1, 1
        )  # array of scenario demands needs to be reshaped if demand is a number
    else:
        X = np.array([all_scens[s][dkey] for s in all_scens.keys()])

    kmeans = KMeans(n_clusters=num_centers, random_state=0, n_init="auto").fit(
        X
    )  # find bundle centers
    centers = kmeans.cluster_centers_  # list of bundle centers

    max_diffs = {}
    for s in all_scens:  # map scenarios to furthest bundle center
        diffs = [
            float(np.linalg.norm(centers[i] - all_scens[s][dkey]))
            for i in range(len(centers))
        ]
        max_diffs[s] = diffs.index(max(diffs))

    bundle = {}
    for model in models:
        for s in data[model]:
            if (
                f"bundle_{max_diffs[s]}" in bundle
            ):  # ensures empty bundles aren't created if centers have no mapped scenarios
                bundle[f"bundle_{max_diffs[s]}"]["scenarios"].update(
                    {bf.scen_key(model, s): data[model][s][pkey]}
                )
            else:
                bundle[f"bundle_{max_diffs[s]}"] = dict(
                    scenarios={bf.scen_key(model, s): data[model][s][pkey]},
                    Probability=0,
                )
    # bundle probability is normalized sum of scenario probabilities within bundle
    for bkey in bundle.keys():
        bundle[bkey]["Probability"] = sum(bundle[bkey]["scenarios"].values())
    # normalize bundle probabilities
    bundle_prob_norm = sum(bundle[bkey]["Probability"] for bkey in bundle.keys())
    for bkey in bundle.keys():
        bundle[bkey]["Probability"] *= 1 / bundle_prob_norm

    # normalization term for if some centers have no mapped scenarios
    fewer_centers_norm = sum(bundle[bkey]["Probability"] for bkey in bundle.keys())
    for bkey in bundle.keys():
        bundle[bkey]["Probability"] *= 1 / fewer_centers_norm
        norm_factor = sum(
            bundle[bkey]["scenarios"].values()
        )  # normalizing bundle probabilities
        for skey in bundle[bkey]["scenarios"].keys():
            bundle[bkey]["scenarios"][skey] *= 1 / norm_factor

    return bundle


def sf_random(data, models=None, bundle_args=None):
    """
    Scenarios are randomly grouped into bundles of approx. same size
        - Can specify bundle size in bundle_args as "bun_size"
        - Bundle size will default to 2-3 scenarios per bundle by default
    """

    if models is None:
        models = list(data.keys())

    model0 = models[0]
    pkey = bf.check_data_dict_keys(data, model0, bundle_args)[1]

    scens = {sname: sval for model in models for (sname, sval) in data[model].items()}

    # user can optionally set random seed
    seed_value = 972819128347298
    if bundle_args != None:
        seed_value = bundle_args.get("seed", seed_value)
        num_buns = bundle_args.get(
            "num_buns", (-len(scens) // -2)
        )  # defaults to 2-3 scens per bundle
    random.seed(seed_value)

    if bundle_args == None:
        num_buns = -len(scens) // -2  # defaults to 2-3 scens per bundle

    if num_buns > len(scens):  # can't construct more bundles than num. of scenarios
        raise RuntimeError(f"Number of bundles must be <= number of scenarios")

    if num_buns < 0:  # number of bundles must be positive
        raise ValueError(f"Number of bundles must be positive")

    if (int(num_buns) == num_buns) == False:  # number of bundles must be integer
        raise ValueError(f"Number of bundles must be integer")

    base_bunsize = len(scens) // num_buns  # approximate bundle size for each bundle
    rem = len(scens) % num_buns  # remainder of |scens|/num_buns

    """
    if the number of scenarios is not divisible by the number of bundles, some bundles
    will have size base_bunsize + 1
    """
    bunsize = [None] * num_buns
    for i in range(len(bunsize)):
        bunsize[i] = base_bunsize
    for i in range(rem):
        bunsize[i] += 1

    for i in range(len(bunsize)):  # check that bundle sizes specified for each bundle
        if any(bun == None for bun in bunsize):
            raise RuntimeError(f"No bundle size specified for bundle {i}")

    scen_idx = []  # temporary list
    for model in models:
        for skey in data[
            model
        ]:  # using indices rather than IDs to allow for non-numeric scenario IDs
            scen_idx.append(bf.scen_key(model, skey))

    temp_bundle = {}  # randomly assign scenarios to bundles
    for bun_idx in range(num_buns):
        temp_list = random.sample(scen_idx, bunsize[bun_idx])
        temp_bundle[bun_idx] = temp_list
        for temp_scen_idx in temp_list:
            scen_idx.remove(temp_scen_idx)
    if len(scen_idx) != 0:  # check that each scenario is assigned to a bundle
        raise RuntimeError(f"Scenarios {scen_idx} are not assigned to a bundle")

    bundle = {}
    for bun_idx in range(num_buns):
        bun_prob = sum(
            scens[f"{temp_bundle[bun_idx][l][1]}"][pkey]
            for l, _ in enumerate(temp_bundle[bun_idx])
        )
        bundle[f"rand_{bun_idx}"] = {
            "scenarios": {
                temp_bundle[bun_idx][l]: scens[f"{temp_bundle[bun_idx][l][1]}"][pkey]
                / bun_prob
                for l, _ in enumerate(temp_bundle[bun_idx])
            },
            "Probability": bun_prob,
        }

    return bundle

