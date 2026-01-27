import random
from sklearn.cluster import KMeans
import numpy as np
import numbers

import sparow.logs
import sparow.sp.bundling.bundling_helper_functions as bf

logger = sparow.logs.logger

"""
******************* MULTI-FIDELITY SCHEMES *******************
"""


def mf_bundle_from_list(data, models=None, model_weight=None, bundle_args=None):
    """
    This scheme accepts a list of lists provided by the user, where each inner list is a bundle.
    Assumes user will pass in the list of lists to bundle_args with key "bundles",
        e.g., bundle_args["bundles"] = [[(HF, scen_1), (LF, scen_3)], [(HF, scen_0), (LF, scen_2)]].
    This scheme is well-suited for users who wish to specify their own bundles.
    """
    if bundle_args:
        list_of_bundles = bundle_args.get("bundles")
    else:
        list_of_bundles = None
    if list_of_bundles is None or len(list_of_bundles) == 0:
        raise RuntimeError(f"mf_bundle_from_list scheme requires 'bundles' key")

    if models is None:
        models = list(data.keys())
    assert (
        len(models) > 1
    ), "Expecting multiple models for mf_bundle_from_list; see bundle_from_list for equivalent single fidelity scheme"

    model0 = models[0]  # the first model in models is assumed to be the HF model
    pkey = bf.check_data_dict_keys(data, model0, bundle_args)[1]

    if pkey is None:
        raise RuntimeError(f"Specify probability_key in bundle_args")

    bundle = {}
    for bundle_list_idx, bundle_list in enumerate(list_of_bundles):
        bundle[f"bundle_{bundle_list_idx}"] = dict(
            scenarios={
                bundle_list[b_tuple_idx]: data[f"{b_tuple[0]}"][f"{b_tuple[1]}"][pkey]
                for b_tuple_idx, b_tuple in enumerate(bundle_list)
            },
            Probability=sum(
                data[f"{b_tuple[0]}"][f"{b_tuple[1]}"][pkey] for b_tuple in bundle_list
            )
            / len(data.keys()),
        )

    # normalize scenario probabilities within bundles
    for b_key in bundle.keys():
        norm_factor = sum(bundle[b_key]["scenarios"].values())
        for s_key in bundle[b_key]["scenarios"]:
            bundle[b_key]["scenarios"][s_key] *= 1 / norm_factor

    return bundle


def mf_kmeans_similar(data, models=None, model_weight=None, bundle_args=None):
    """
    LF scenarios are bundled with closest HF scenario; HF scenarios are bundle centers
        - bun_size (approx. size of each bundle) can be passed into bundle_args. default is 2.
        - ensure there are no duplicate/redundant scenarios before using this!
        - all scenarios must have unique names!
        - ignores model weights for now!!!
    """

    if models is None:
        models = list(data.keys())
    assert (
        len(models) > 1
    ), "Expecting multiple models for mf_kmeans_similar; see kmeans_similar for equivalent single fidelity scheme"

    model0 = models[0]  # the first model in models is assumed to be the HF model
    dkey = bf.check_data_dict_keys(data, model0, bundle_args, dkey_required=True)[0]
    pkey = bf.check_data_dict_keys(data, model0, bundle_args, dkey_required=True)[1]

    if pkey is None:
        raise RuntimeError(f"Specify probability_key in bundle_args")

    if bundle_args is not None:  # default bundle size is 2
        bun_size = bundle_args.get("bun_size", 2)
    else:
        bun_size = 2

    num_scens = sum(len(data[model]) for model in models)  # total number of scenarios
    if bun_size > num_scens:
        raise ValueError(f"Bundle size cannot exceed number of scenarios")

    num_centers = len(
        data[model0]
    )  # number of bundle centers (NOT NECESSARILY the same as number of bundles!!)
    all_scens = {
        sname: sval for model in models for (sname, sval) in data[model].items()
    }
    hf_scens = {sname: sval for (sname, sval) in data[model0].items()}
    lf_scens = {
        sname: sval for model in models[1:] for (sname, sval) in data[model].items()
    }

    if isinstance(data[model0][next(iter(all_scens))][dkey], numbers.Number) == True:
        arr = np.array([data[model0][skey][dkey] for skey in data[model0].keys()])
        X = arr.reshape(
            -1, 1
        )  # array of scenario demands needs to be reshaped if demand is 1-dimensional
    else:
        X = np.array([data[model0][skey][dkey] for skey in data[model0].keys()])
    # only HF scenarios are used to determine bundle centers
    kmeans = KMeans(n_clusters=num_centers, random_state=0, n_init="auto").fit(
        X
    )  # find bundle centers
    centers = kmeans.cluster_centers_  # list of bundle centers
    # in theory, more than 1 HF scenario can belong in a single bundle
    diffs = {}
    for hs in hf_scens:  # map scenarios to closest bundle center
        dem_diffs = [
            float(np.linalg.norm(centers[i] - hf_scens[hs][dkey]))
            for i in range(len(centers))
        ]
        diffs[hs] = dem_diffs.index(min(dem_diffs))
    for ls in lf_scens:  # map LF scenarios to closest bundle center
        dem_diffs = [
            float(np.linalg.norm(centers[i] - lf_scens[ls][dkey]))
            for i in range(len(centers))
        ]
        diffs[ls] = dem_diffs.index(min(dem_diffs))

    bundle = {}
    for model in models:
        for s in data[model]:
            if model_weight:
                if (
                    f"bundle_{centers[diffs[s]][0]}" in bundle
                ):  # ensures empty bundles aren't created if centers have no mapped scenarios
                    bundle[f"bundle_{centers[diffs[s]][0]}"]["scenarios"].update(
                        {
                            bf.scen_key(model, s): model_weight[model]
                            * data[model][s][pkey]
                        }
                    )
                else:
                    bundle[f"bundle_{centers[diffs[s]][0]}"] = dict(
                        scenarios={
                            bf.scen_key(model, s): model_weight[model]
                            * data[model][s][pkey]
                        },
                        Probability=0,
                    )
            else:
                if (
                    f"bundle_{centers[diffs[s]][0]}" in bundle
                ):  # ensures empty bundles aren't created if centers have no mapped scenarios
                    bundle[f"bundle_{centers[diffs[s]][0]}"]["scenarios"].update(
                        {bf.scen_key(model, s): data[model][s][pkey]}
                    )
                else:
                    bundle[f"bundle_{centers[diffs[s]][0]}"] = dict(
                        scenarios={bf.scen_key(model, s): data[model][s][pkey]},
                        Probability=0,
                    )
    # scenario probabilities within each bundle are normalized for model_weight
    if model_weight:
        for bkey in bundle.keys():
            num_scens_per_fidelity = {model: 0 for model in models}
            for (mod, _), _ in bundle[bkey]["scenarios"].items():
                num_scens_per_fidelity[mod] += 1
            bundle_weight_factor = sum(
                model_weight[model] * num_scens_per_fidelity[model] for model in models
            )
            for skey in bundle[bkey]["scenarios"]:
                bundle[bkey]["scenarios"][skey] *= 1 / bundle_weight_factor
    # sum scenario probabilities within bundle to obtain pre-normalized bundle probability
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


def mf_kmeans_dissimilar(data, models=None, model_weight=None, bundle_args=None):
    """
    LF scenarios are bundled with furthest HF scenario; HF scenarios are bundle centers
        - bun_size (approx. size of each bundle) can be passed into bundle_args. default is 2.
        - ensure there are no duplicate/redundant scenarios before using this!
        - all scenarios must have unique names!
        - ignores model weights for now!!!
    """

    if models is None:
        models = list(data.keys())
    assert (
        len(models) > 1
    ), "Expecting multiple models for mf_kmeans_dissimilar; see kmeans_dissimilar for equivalent single fidelity scheme"

    model0 = models[0]  # the first model in models is assumed to be the HF model
    dkey = bf.check_data_dict_keys(data, model0, bundle_args, dkey_required=True)[0]
    pkey = bf.check_data_dict_keys(data, model0, bundle_args, dkey_required=True)[1]

    if pkey is None:
        raise RuntimeError(f"Specify probability_key in bundle_args")

    if bundle_args is not None:  # default bundle size is 2
        bun_size = bundle_args.get("bun_size", 2)
    else:
        bun_size = 2

    num_scens = sum(len(data[model]) for model in models)  # total number of scenarios
    if bun_size > num_scens:
        raise ValueError(f"Bundle size cannot exceed number of scenarios")

    num_centers = len(
        data[model0]
    )  # number of bundle centers (NOT NECESSARILY the same as number of bundles!!)
    all_scens = {
        sname: sval for model in models for (sname, sval) in data[model].items()
    }
    hf_scens = {sname: sval for (sname, sval) in data[model0].items()}
    lf_scens = {
        sname: sval for model in models[1:] for (sname, sval) in data[model].items()
    }

    if isinstance(data[model0][next(iter(all_scens))][dkey], numbers.Number) == True:
        arr = np.array([data[model0][skey][dkey] for skey in data[model0].keys()])
        X = arr.reshape(
            -1, 1
        )  # array of scenario demands needs to be reshaped if demand is 1-dimensional
    else:
        X = np.array([data[model0][skey][dkey] for skey in data[model0].keys()])
    # only HF scenarios are used to determine bundle centers
    kmeans = KMeans(n_clusters=num_centers, random_state=0, n_init="auto").fit(
        X
    )  # find bundle centers
    centers = kmeans.cluster_centers_  # list of bundle centers
    # in theory, more than 1 HF scenario can belong in a single bundle
    diffs = {}
    for hs in hf_scens:  # map HF scenarios to closest bundle center
        dem_diffs = [
            float(np.linalg.norm(centers[i] - hf_scens[hs][dkey]))
            for i in range(len(centers))
        ]
        diffs[hs] = dem_diffs.index(min(dem_diffs))
    for ls in lf_scens:  # map LF scenarios to furthest bundle center
        dem_diffs = [
            float(np.linalg.norm(centers[i] - lf_scens[ls][dkey]))
            for i in range(len(centers))
        ]
        diffs[ls] = dem_diffs.index(max(dem_diffs))

    bundle = {}
    for model in models:
        for s in data[model]:
            if model_weight:
                if (
                    f"bundle_{centers[diffs[s]][0]}" in bundle
                ):  # ensures empty bundles aren't created if centers have no mapped scenarios
                    bundle[f"bundle_{centers[diffs[s]][0]}"]["scenarios"].update(
                        {
                            bf.scen_key(model, s): model_weight[model]
                            * data[model][s][pkey]
                        }
                    )
                else:
                    bundle[f"bundle_{centers[diffs[s]][0]}"] = dict(
                        scenarios={
                            bf.scen_key(model, s): model_weight[model]
                            * data[model][s][pkey]
                        },
                        Probability=0,
                    )
            else:
                if (
                    f"bundle_{centers[diffs[s]][0]}" in bundle
                ):  # ensures empty bundles aren't created if centers have no mapped scenarios
                    bundle[f"bundle_{centers[diffs[s]][0]}"]["scenarios"].update(
                        {bf.scen_key(model, s): data[model][s][pkey]}
                    )
                else:
                    bundle[f"bundle_{centers[diffs[s]][0]}"] = dict(
                        scenarios={bf.scen_key(model, s): data[model][s][pkey]},
                        Probability=0,
                    )
    # scenario probabilities within each bundle are normalized for model_weight
    if model_weight:
        for bkey in bundle.keys():
            num_scens_per_fidelity = {model: 0 for model in models}
            for (mod, _), _ in bundle[bkey]["scenarios"].items():
                num_scens_per_fidelity[mod] += 1
            bundle_weight_factor = sum(
                model_weight[model] * num_scens_per_fidelity[model] for model in models
            )
            for skey in bundle[bkey]["scenarios"]:
                bundle[bkey]["scenarios"][skey] *= 1 / bundle_weight_factor
    # sum scenario probabilities within bundle to obtain pre-normalized bundle probability
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


## TODO: remove(???) -R
def similar_partitions(data, models=None, model_weight=None, bundle_args=None):
    """
    Each HF scenario bundled with closest LF scenario (all LF scenarios not necessarily used)
        - Each bundle contains exactly 1 HF and 1 LF scenario
        - LF scenarios may be repeated across bundles
        - currently only works with 1 LF model!
        - Can have different numbers of HF and LF scenarios
    """
    if models is None:
        models = list(data.keys())
    assert len(models) > 1, "Expecting multiple models for similar_partitions"

    """
    Distance metric is user-defined/application-specific
    """
    distance_function = bundle_args["distance_function"]

    model0 = models[0]  # the first model in models is assumed to be the HF model
    pkey = bf.check_data_dict_keys(data, model0, bundle_args)[1]

    if pkey is None:
        raise RuntimeError(f"Specify probability_key in bundle_args")

    HFscenarios = list(data[model0].keys())  # list of HF scenario names
    LFscenarios = {}  # all other models are LF
    for model in models[1:]:
        LFscenarios[model] = list(data[model].keys())  # list of LF scenario names

    demand_diffs = distance_function(
        data, models
    )  # distance between each HF/LF scenario
    HFmap = {}  # keys are HF scenario, values are closest LF scenario
    for model in models[1:]:
        for hs_ind, hs in enumerate(HFscenarios):
            min_key = None
            min_value = float("inf")
            for ls_ind, ls in enumerate(LFscenarios[model]):
                if (
                    demand_diffs[(hs_ind, ls_ind)] < min_value
                    and demand_diffs[(hs_ind, ls_ind)] != 0
                ):
                    min_key = ls
                    min_value = demand_diffs[(hs_ind, ls_ind)]
            HFmap[hs] = [bf.scen_key(model, min_key)]

    #
    # Create the final bundle object
    #
    bundle = {}
    for hs in HFscenarios:  # each HF scenario belongs to its own bundle
        if model_weight:
            bundle[f"{model0}_{hs}"] = dict(
                scenarios={
                    bf.scen_key(model0, hs): model_weight[model0]
                    * data[model0][hs][pkey]
                },
                Probability=1
                / len(
                    HFscenarios
                ),  ###################### assuming uniform probs for now!!!!!!!!!!!!!!!!!
            )
        else:
            bundle[f"{model0}_{hs}"] = dict(
                scenarios={bf.scen_key(model0, hs): data[model0][hs][pkey]},
                Probability=1
                / len(
                    HFscenarios
                ),  ###################### assuming uniform probs for now!!!!!!!!!!!!!!!!!,
            )
        for ls in HFmap[
            hs
        ]:  # map the LF scenarios that are used to corresponding bundle
            if model_weight:
                bundle[f"{model0}_{hs}"]["scenarios"][bf.scen_key(ls[0], ls[1])] = (
                    model_weight[ls[0]] * data[ls[0]][ls[1]][pkey]
                )
            else:
                bundle[f"{model0}_{hs}"]["scenarios"][bf.scen_key(ls[0], ls[1])] = data[
                    ls[0]
                ][ls[1]][pkey]
                model_weight = {}
                for model in models:
                    model_weight[model] = 1
        model_weight_factor = sum(model_weight.values())
        for b_key in bundle[f"{model0}_{hs}"]["scenarios"].keys():
            bundle[f"{model0}_{hs}"]["scenarios"][b_key] *= 1 / model_weight_factor
        norm_factor = sum(bundle[f"{model0}_{hs}"]["scenarios"].values())
        for b_key in bundle[f"{model0}_{hs}"]["scenarios"].keys():
            bundle[f"{model0}_{hs}"]["scenarios"][b_key] *= 1 / norm_factor

    return bundle


## TODO: remove(???) -R
def dissimilar_partitions(data, models=None, model_weight=None, bundle_args=None):
    """
    Each HF scenario bundled with furthest LF scenario (all LF scenarios not necessarily used)
        - Each bundle contains exactly 1 HF and 1 LF scenario
        - LF scenarios may be repeated across bundles
        - USING MODEL WEIGHTS WILL IGNORE THE SCENARIO PROBABILITIES
        - currently only works with 1 LF model!
        - Can have different numbers of HF and LF scenarios
    """
    if models is None:
        models = list(data.keys())
    assert len(models) > 1, "Expecting multiple models for similar_partitions"

    # TODO: return error if bundle_args['distance_function'] is None

    """
    Distance metric is user-defined/application-specific
    """
    distance_function = bundle_args["distance_function"]

    model0 = models[0]  # the first model in models is assumed to be the HF model
    pkey = bf.check_data_dict_keys(data, model0, bundle_args)[1]

    if pkey is None:
        raise RuntimeError(f"Specify probability_key in bundle_args")

    HFscenarios = list(data[model0].keys())
    LFscenarios = {}  # all other models are LF
    for model in models[1:]:
        LFscenarios[model] = list(data[model].keys())

    demand_diffs = distance_function(
        data, models
    )  # distance between each HF/LF scenario
    HFmap = {}  # keys are HF scenario, values are furthest LF scenario
    for model in models[1:]:
        for hs_ind, hs in enumerate(HFscenarios):
            max_key = None
            max_value = 0
            for ls_ind, ls in enumerate(LFscenarios[model]):
                if demand_diffs[(hs_ind, ls_ind)] > max_value:
                    max_key = ls
                    max_value = demand_diffs[(hs_ind, ls_ind)]
            HFmap[hs] = [bf.scen_key(model, max_key)]

    #
    # Create the final bundle object
    #
    bundle = {}
    for hs in HFscenarios:  # each HF scenario belongs to its own bundle
        if model_weight:
            bundle[f"{model0}_{hs}"] = dict(
                scenarios={
                    bf.scen_key(model0, hs): model_weight[model0]
                    * data[model0][hs][pkey]
                },
                Probability=1 / len(HFscenarios),
            )
        else:
            bundle[f"{model0}_{hs}"] = dict(
                scenarios={bf.scen_key(model0, hs): data[model0][hs][pkey]},
                Probability=1 / len(HFscenarios),
            )
        for ls in HFmap[
            hs
        ]:  # map the LF scenarios that are used to corresponding bundle
            if model_weight:
                bundle[f"{model0}_{hs}"]["scenarios"][bf.scen_key(ls[0], ls[1])] = (
                    model_weight[ls[0]] * data[ls[0]][ls[1]][pkey]
                )
            else:
                bundle[f"{model0}_{hs}"]["scenarios"][bf.scen_key(ls[0], ls[1])] = data[
                    ls[0]
                ][ls[1]][pkey]
                model_weight = {}
                for model in models:
                    model_weight[model] = 1
        model_weight_factor = sum(
            model_weight[model0] + model_weight[model] * len(HFmap[hs])
            for model in models[1:]
        )
        for b_key in bundle[f"{model0}_{hs}"]["scenarios"].keys():
            bundle[f"{model0}_{hs}"]["scenarios"][b_key] *= 1 / model_weight_factor
        norm_factor = sum(bundle[f"{model0}_{hs}"]["scenarios"].values())
        for b_key in bundle[f"{model0}_{hs}"]["scenarios"].keys():
            bundle[f"{model0}_{hs}"]["scenarios"][b_key] *= 1 / norm_factor

    return bundle


## TODO: remove(???) -R
def mf_paired(data, models=None, model_weight=None, bundle_args=None):
    """
    Scenarios are paired according to their models
        - Note that scenario probabilities specified for each model are ignored
        - All models must have the same scenario keys
        - Must have same number of HF and LF scenarios
        - Scenarios are not repeated (each scenario belongs to exactly 1 bundle)
    """
    if models is None:
        models = list(data.keys())
    assert len(models) > 1, "Expecting multiple models for mf_paired"

    model0 = models[0]  # the first model in models is assumed to be the HF model
    scenarios = set(data[model0].keys())
    for model in models[1:]:
        assert scenarios == set(
            data[model].keys()
        ), "All models have the same scenario keys"
    #
    # Bundle the paired scenarios for all models
    #
    bundle = {}
    for s in scenarios:
        bundle[f"{s}"] = dict(
            scenarios={bf.scen_key(model, s): 1.0 / len(models) for model in models},
            Probability=1.0 / len(scenarios),
        )

    return bundle


## TODO: remove(???) -R
def mf_random_nested(data, models, model_weight=None, bundle_args=None):
    """
    Bundle randomly selected scenarios for all but the first model
        - All scenario IDs are required to be the same here, and this is used to prevent the
        repeated selection of the scenario from the "high fidelity" model (a.k.. models[0])
        - Must have same number of LF and HF scenarios
        - Note that scenario probabilities specified for each model are ignored.
    """
    if models is None:
        models = list(data.keys())
    assert len(models) > 1, "Expecting multiple models for mf_paired_random"
    #
    # This bundling strategy requires that all models have the same scenario keys.
    #
    model0 = models[0]  # the first model in models is assumed to be the HF model
    bundleIDs = sorted(data[model0].keys())
    for model in models[1:]:
        assert sorted(bundleIDs) == sorted(
            list(data[model].keys())
        ), "All models have the same scenario keys"
    #
    # Process bundle_args
    #
    n = {model: 1 for model in models}
    if bundle_args != None:
        if "seed" in bundle_args:
            random.seed(bundle_args["seed"])
        for model in models:
            if model in bundle_args:
                n[model] = bundle_args[model]

    bundle_scen = {}
    # model[0]
    for b in bundleIDs:
        bundle_scen[b] = {bf.scen_key(model0, b): model_weight[model0]}

    # model[i]
    N = len(bundleIDs)
    scenario_keys = list(sorted(bundleIDs))
    for model in models[1:]:
        #
        # We randomly select n[model] scenarios for the each model 'model'.
        #
        # This uses a single shuffle of all of the scenario keys to encourage a
        # diverse set of scenarios.  We bias sampling by disallowing the scenario
        # use for model0 to be repeated.
        #
        index = list(range(N))
        random.shuffle(index)
        for i, s in enumerate(scenario_keys):
            k = 0
            count = 0
            while count < n[model]:
                i_ = (index[i] + k) % N
                k += 1
                if scenario_keys[i_] == s:
                    continue
                s_ = scenario_keys[i_]
                bundle_scen[s][bf.scen_key(model, s_)] = model_weight[model]
                count += 1
    #
    # Create the final bundle object
    #
    bundle = {}
    for b in bundleIDs:
        total_weight = sum(weight for weight in bundle_scen[b].values())
        for k, w in bundle_scen[b].items():
            bundle_scen[b][k] = w / total_weight
        bundle[f"{model0}_{b}"] = dict(
            scenarios=bundle_scen[b], Probability=1.0 / len(bundleIDs)
        )

    return bundle


def mf_random(data, models, model_weight=None, bundle_args=None):
    """
    Bundle randomly selected scenarios for all but the first model
        - Note that scenario probabilities specified for each model are ignored
        - Each scenario is in exactly 1 bundle
        - Can have different numbers of HF and LF scenarios
    """
    print("here", models)
    if models is None:
        models = list(data.keys())
    assert len(models) > 1, "Expecting multiple models for mf_random"
    print(models)
    model0 = models[0]  # the first model in models is assumed to be the HF model
    bundleIDs = sorted(data[model0].keys())

    """
    Process bundle_args
        - By default, each bundle contains 1 scenario from each LF model
        - Can specify how many scenarios from each LF model are in a bundle in bundle_args
    """
    n = {model: 1 for model in models}
    if bundle_args != None:
        if "seed" in bundle_args:
            random.seed(bundle_args["seed"])
        for model in models:
            if model in bundle_args:
                n[model] = bundle_args[model]

    bundle_scen = {}
    # model[0]
    for b in bundleIDs:  # each bundle contains 1 HF scenario
        bundle_scen[b] = {bf.scen_key(model0, b): model_weight[model0]}

    # model[i]
    for b in bundleIDs:
        for model in models[1:]:
            #
            # We randomly select n[model] scenarios for the each model 'model'.
            #
            # This uses a single shuffle of all of the scenario keys to encourage a
            # diverse set of scenarios.
            #
            N = len(data[model])  # number of scenarios for model
            scenario_keys = list(sorted(data[model].keys()))
            index = list(range(N))
            random.shuffle(index)
            k = 0
            while k < min(N, n[model]):  # randomly assign LF scenario to bundle
                s_ = scenario_keys[index[k]]
                bundle_scen[b][bf.scen_key(model, s_)] = model_weight[model]
                k += 1
    #
    # Create the final bundle object
    #
    bundle = {}
    for b in bundleIDs:
        total_weight = sum(weight for weight in bundle_scen[b].values())
        for k, w in bundle_scen[b].items():
            bundle_scen[b][k] = w / total_weight
        bundle[f"{model0}_{b}"] = dict(
            scenarios=bundle_scen[b], Probability=1.0 / len(bundleIDs)
        )

    return bundle
