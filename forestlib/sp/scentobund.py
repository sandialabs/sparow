import json
import munch
import random
import types
from sklearn.cluster import KMeans
import numpy as np
import numbers

"""
* specify which bundling scheme (function) is used via "bundle_scheme" in sp.py

* bundle is a dictionary of dictionaries
    - keys are names of bundles
    - each key corresponds to a dictionary, with keys:
        - 'scenarios' (i.e., which scenarios are in the bundle)
            - 'scenarios' is itself a dictionary containing keys: (model fidelity, scenario name) pairs,
               and values: model fidelity weight (if model_weight=True) or scenario probability otherwise
        - 'Probability' (the bundle probability)

* example of a multi-fidelity bundle:
    {
    "bundle_0": {
                "scenarios": {
                    ("HF", "scen_0"): 0.6,
                    ("LF", "scen_2"): 0.25,
                    ("LF", "scen_3"): 0.15,
                },
                "Probability": 0.4,
            },
    "bundle_1": {"scenarios": {
                    ("HF", "scen_1"): 0.7
                    ("LF", "scen_4"): 0.3,
                }, 
                "Probability": 0.6
            },
    }

* it is assumed that the data containing the scenario information is a dictionary
    - keys are model names (e.g., 'HF', 'LF')
    - values are themselves dictionaries
        - keys are scenario names (e.g., 'scen_0', 'scen_1')
        - values are themselves dictionaries
            - keys are scenario information labels (e.g., 'Demand', 'Probability')
            - values are corresponding numerical values or lists

# example of data dictionary:
    {
        "HF": {
            "scen_1": {
                "Demand": [3, 2],
                "Probability": 0.4,
            },
            "scen_0": {
                "Demand": [1, 1],
                "Probability": 0.6,
            },
        },
        "LF": {
            "scen_3": {
                "Demand": [4, 4],
                "Probability": 0.2,
            },
            "scen_2": {
                "Demand": [2, 2],
                "Probability": 0.8,
            },
        },
    }
"""


"""
******************* HELPER FUNCTIONS *******************
"""


def JSdecoded(item: dict, dict_key=False):
    if isinstance(item, list):
        return [JSdecoded(e) for e in item]
    elif isinstance(item, dict):
        return {literal_eval(key): value for key, value in item.items()}
    return item


def JSencoded(item, dict_key=False):
    if isinstance(item, tuple):
        if dict_key:
            return str(item)
        else:
            return list(item)
    elif isinstance(item, list):
        return [JSencoded(e) for e in item]
    elif isinstance(item, dict):
        return {JSencoded(key, True): JSencoded(value) for key, value in item.items()}
    elif isinstance(item, set):
        return list(item)
    elif type(item) is types.FunctionType:
        return None
    return item


def scen_name(model, scenario):
    if model is None:
        return f"{scenario}"
    return f"{model}_{scenario}"


def scen_key(model, scenario):
    return (model, scenario)


"""
******************* MULTI-FIDELITY SCHEMES *******************
"""

### THIS SCHEME ISN'T WORKING YET, DO NOT USE! TODO: fix this -R
def mf_kmeans_similar(data, model_weight, models=None, bundle_args=None):
    """
    LF scenarios are bundled with closest HF scenario; HF scenarios are bundle centers
        - bun_size (approx. size of each bundle) can be passed into bundle_args. default is 2.
        - ensure there are no duplicate/redundant scenarios before using this! 
        - all scenarios must have unique names! 
    """
    if models is None:
        models = list(data.keys())

    if "bun_size" in bundle_args:     # default bundle size is 2
        bun_size = bundle_args['bun_size']
    else:
        bun_size = 2

    num_scens = sum(len(data[model]) for model in models)   # total number of scenarios
    if bun_size > num_scens:
        raise ValueError(f"Bundle size cannot exceed number of scenarios")

    num_centers = -(num_scens// -bun_size)  # number of bundle centers (NOT NECESSARILY the same as number of bundles!!)
    all_scens = {sname: sval for model in models for (sname, sval) in data[model].items()}

    if isinstance(data[models[0]][next(iter(all_scens))]['Demand'], numbers.Number) == True:
        arr = np.array([all_scens[s]['Demand'] for s in all_scens.keys()])
        X = arr.reshape(-1,1)   # array of scenario demands needs to be reshaped if demand is a number
    else:
        X = np.array([all_scens[s]['Demand'] for s in all_scens.keys()])

    kmeans = KMeans(n_clusters=num_centers, random_state=0, n_init="auto").fit(X)   # find bundle centers
    s_assign = kmeans.labels_   # list of closest bundle centers
    sbmap = {s: int(s_assign[ind_s]) for ind_s, s in enumerate(iter(all_scens))}    # map scenarios to closest bundle center

    bundle = {}
    for model in models:
        for s in data[model]:
            if f"bundle_{sbmap[s]}" in bundle:  # ensures empty bundles aren't created if centers have no mapped scenarios
                bundle[f"bundle_{sbmap[s]}"]["scenarios"].update({scen_key(model, s): data[model][s]['Probability']})
            else:
                bundle[f"bundle_{sbmap[s]}"] = dict(
                    scenarios = {scen_key(model, s): data[model][s]['Probability']},
                    Probability = 1/num_centers
                )

    # normalization term for if some centers have no mapped scenarios
    fewer_centers_norm = sum(bundle[bkey]["Probability"] for bkey in bundle.keys())
    for bkey in bundle.keys():
        bundle[bkey]["Probability"] *= (1/fewer_centers_norm)
        norm_factor = sum(bundle[bkey]["scenarios"].values())   # normalizing bundle probabilities
        for skey in bundle[bkey]["scenarios"].keys():
            bundle[bkey]["scenarios"][skey] *= (1/norm_factor)

    return bundle


def similar_partitions(data, model_weight=None, models=None, bundle_args=None):
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
    HFscenarios = list(data[model0].keys())
    LFscenarios = {}  # all other models are LF
    for model in models[1:]:
        LFscenarios[model] = list(data[model].keys())

    demand_diffs = distance_function(
        data, models
    )  # distance between each HF/LF scenario
    HFmap = {}  # keys are HF scenario, values are closest LF scenario
    for model in models[1:]:
        for hs_ind, hs in enumerate(HFscenarios):
            min_key = None
            min_value = float('inf')
            for ls_ind, ls in enumerate(LFscenarios[model]):
                if demand_diffs[(hs_ind,ls_ind)] < min_value and demand_diffs[(hs_ind,ls_ind)] != 0:
                    min_key = ls
                    min_value = demand_diffs[(hs_ind,ls_ind)]
            HFmap[hs] = [scen_key(model, min_key)]

    #
    # Create the final bundle object
    #
    bundle = {}
    for hs in HFscenarios:  # each HF scenario belongs to its own bundle
        if model_weight:
            bundle[f"{model0}_{hs}"] = dict(
                scenarios={scen_key(model0, hs): model_weight[model0]*data[model0][hs]['Probability']},
                Probability=1 / len(HFscenarios), ###################### assuming uniform probs for now!!!!!!!!!!!!!!!!!
            )
        else:
            bundle[f"{model0}_{hs}"] = dict(
                scenarios={scen_key(model0, hs): data[model0][hs]["Probability"]},
                Probability=1 / len(HFscenarios), ###################### assuming uniform probs for now!!!!!!!!!!!!!!!!!,
        )
        for ls in HFmap[
            hs
        ]:  # map the LF scenarios that are used to corresponding bundle
            if model_weight:
                bundle[f"{model0}_{hs}"]["scenarios"][scen_key(ls[0], ls[1])] = (
                    model_weight[ls[0]]*data[ls[0]][ls[1]]["Probability"])
            else:
                bundle[f"{model0}_{hs}"]["scenarios"][scen_key(ls[0], ls[1])] = data[
                    ls[0]
                ][ls[1]]["Probability"]
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


def dissimilar_partitions(data, model_weight=None, models=None, bundle_args=None):
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
                if demand_diffs[(hs_ind,ls_ind)] > max_value:
                    max_key = ls
                    max_value = demand_diffs[(hs_ind,ls_ind)]
            HFmap[hs] = [scen_key(model, max_key)]

    #
    # Create the final bundle object
    #
    bundle = {}
    for hs in HFscenarios:  # each HF scenario belongs to its own bundle
        if model_weight:
            bundle[f"{model0}_{hs}"] = dict(
                scenarios={scen_key(model0, hs): model_weight[model0]*data[model0][hs]['Probability']},
                Probability=1 / len(HFscenarios),
            )
        else:
            bundle[f"{model0}_{hs}"] = dict(
                scenarios={scen_key(model0, hs): data[model0][hs]["Probability"]},
                Probability=1 / len(HFscenarios),
            )
        for ls in HFmap[
            hs
        ]:  # map the LF scenarios that are used to corresponding bundle
            if model_weight:
                bundle[f"{model0}_{hs}"]["scenarios"][scen_key(ls[0], ls[1])] = (
                    model_weight[ls[0]]*data[ls[0]][ls[1]]["Probability"]
                )
            else:
                bundle[f"{model0}_{hs}"]["scenarios"][scen_key(ls[0], ls[1])] = data[
                    ls[0]
                ][ls[1]]["Probability"]
                model_weight = {}
                for model in models:
                    model_weight[model] = 1
        model_weight_factor = sum(model_weight[model0] + model_weight[model]*len(HFmap[hs]) for model in models[1:])
        for b_key in bundle[f"{model0}_{hs}"]["scenarios"].keys():
            bundle[f"{model0}_{hs}"]["scenarios"][b_key] *= 1 / model_weight_factor
        norm_factor = sum(bundle[f"{model0}_{hs}"]["scenarios"].values())
        for b_key in bundle[f"{model0}_{hs}"]["scenarios"].keys():
            bundle[f"{model0}_{hs}"]["scenarios"][b_key] *= 1 / norm_factor

    return bundle


def mf_paired(data, model_weight, models=None, bundle_args=None):
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
            scenarios={scen_key(model, s): 1.0 / len(models) for model in models},
            Probability=1.0 / len(scenarios),
        )

    return bundle


def mf_random_nested(data, model_weight, models, bundle_args=None):
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
    bundleIDs = set(data[model0].keys())
    for model in models[1:]:
        assert bundleIDs == set(
            data[model].keys()
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
        bundle_scen[b] = {scen_key(model0, b): model_weight[model0]}

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
                bundle_scen[s][scen_key(model, s_)] = model_weight[model]
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
            scenarios=bundle_scen[b],
            Probability=1.0 / len(bundleIDs),
        )

    return bundle


def mf_random(data, model_weight, models, bundle_args=None):
    """
    Bundle randomly selected scenarios for all but the first model
        - Note that scenario probabilities specified for each model are ignored
        - Each scenario is in exactly 1 bundle
        - Can have different numbers of HF and LF scenarios
    """
    if models is None:
        models = list(data.keys())
    assert len(models) > 1, "Expecting multiple models for mf_random"

    model0 = models[0]  # the first model in models is assumed to be the HF model
    bundleIDs = set(data[model0].keys())

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
        bundle_scen[b] = {scen_key(model0, b): model_weight[model0]}

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
                bundle_scen[b][scen_key(model, s_)] = model_weight[model]
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
            scenarios=bundle_scen[b],
            Probability=1.0 / len(bundleIDs),
        )

    return bundle


def mf_ordered(data, model_weight=None, models=None, bundle_args=None):
    """
    Scenarios are paired according to their models
        - Each HF scenario in exactly 1 bundle; LF scenarios are added one at a
        time to bundles in order they are listed
        - Can have different numbers of HF and LF scenarios
        - Scenarios are not repeated (each scenario belongs to exactly 1 bundle)
    """
    if models is None:
        models = list(data.keys())
    assert len(models) > 1, "Expecting multiple models for mf_ordered"

    #
    # scenarios from each model are paired by order they appear in scenario list
    #
    model0 = models[0]  # the first model in models is assumed to be the HF model
    HFscenarios = list(data[model0].keys())
    LFscenarios = {}  # all other models are LF
    for model in models[1:]:
        LFscenarios[model] = list(data[model].keys())

    LFmap = (
        {}
    )  # keys are tuples (model, LF scenario) and values are HF scenario to be bundled with
    for model in models[1:]:
        for idx_ls, ls in enumerate(LFscenarios[model]):
            LFmap[(model, ls)] = HFscenarios[idx_ls % len(HFscenarios)]

    #
    # Create the final bundle object
    #
    bundle = {}
    for hs in HFscenarios:  # each HF scenario belongs in exactly 1 bundle
        if model_weight:
            bundle[f"{model0}_{hs}"] = dict(
                scenarios={scen_key(model0, hs): model_weight[model0]},
                Probability=1 / len(HFscenarios),
            )
        else:
            bundle[f"{model0}_{hs}"] = dict(
                scenarios={scen_key(model0, hs): data[model0][hs]["Probability"]},
                Probability=data[model0][hs]["Probability"],
            )
    for model in models[
        1:
    ]:  # each LF scenario bundled with HF scenario based on order in data dict
        for ls in LFscenarios[model]:
            if model_weight:
                bundle[f"{model0}_{LFmap[(model, ls)]}"]["scenarios"][
                    scen_key(model, ls)
                ] = model_weight[model]
            else:
                bundle[f"{model0}_{LFmap[(model, ls)]}"]["scenarios"][
                    scen_key(model, ls)
                ] = data[model][ls]["Probability"]
    for hs in HFscenarios:  # normalize bundle probabilities
        norm_factor = sum(bundle[f"{model0}_{hs}"]["scenarios"].values())
        for b_key in bundle[f"{model0}_{hs}"]["scenarios"].keys():
            bundle[f"{model0}_{hs}"]["scenarios"][b_key] *= 1 / norm_factor

    return bundle


"""
******************* SINGLE-FIDELITY SCHEMES *******************
"""


def single_scenario(data, model_weight, models=None, bundle_args=None):
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


def single_bundle(data, model_weight, models=None, bundle_args=None):
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


def kmeans_similar(data, model_weight, models=None, bundle_args=None):
    """
    Each scenario is paired by closest distance
        - bun_size (approx. size of each bundle) can be passed into bundle_args. default is 2.
        - ensure there are no duplicate/redundant scenarios before using this! 
        - all scenarios must have unique names! 
    """
    if models is None:
        models = list(data.keys())

    if "bun_size" in bundle_args:     # default bundle size is 2
        bun_size = bundle_args['bun_size']
    else:
        bun_size = 2

    num_scens = sum(len(data[model]) for model in models)   # total number of scenarios
    if bun_size > num_scens:
        raise ValueError(f"Bundle size cannot exceed number of scenarios")

    num_centers = -(num_scens// -bun_size)  # number of bundle centers (NOT NECESSARILY the same as number of bundles!!)
    all_scens = {sname: sval for model in models for (sname, sval) in data[model].items()}

    if isinstance(data[models[0]][next(iter(all_scens))]['Demand'], numbers.Number) == True:
        arr = np.array([all_scens[s]['Demand'] for s in all_scens.keys()])
        X = arr.reshape(-1,1)   # array of scenario demands needs to be reshaped if demand is a number
    else:
        X = np.array([all_scens[s]['Demand'] for s in all_scens.keys()])

    kmeans = KMeans(n_clusters=num_centers, random_state=0, n_init="auto").fit(X)   # find bundle centers
    s_assign = kmeans.labels_   # list of closest bundle centers
    sbmap = {s: int(s_assign[ind_s]) for ind_s, s in enumerate(iter(all_scens))}    # map scenarios to closest bundle center

    bundle = {}
    for model in models:
        for s in data[model]:
            if f"bundle_{sbmap[s]}" in bundle:  # ensures empty bundles aren't created if centers have no mapped scenarios
                bundle[f"bundle_{sbmap[s]}"]["scenarios"].update({scen_key(model, s): data[model][s]['Probability']})
            else:
                bundle[f"bundle_{sbmap[s]}"] = dict(
                    scenarios = {scen_key(model, s): data[model][s]['Probability']},
                    Probability = 1/num_centers
                )

    # normalization term for if some centers have no mapped scenarios
    fewer_centers_norm = sum(bundle[bkey]["Probability"] for bkey in bundle.keys())
    for bkey in bundle.keys():
        bundle[bkey]["Probability"] *= (1/fewer_centers_norm)
        norm_factor = sum(bundle[bkey]["scenarios"].values())   # normalizing bundle probabilities
        for skey in bundle[bkey]["scenarios"].keys():
            bundle[bkey]["scenarios"][skey] *= (1/norm_factor)

    return bundle


def kmeans_dissimilar(data, model_weight, models=None, bundle_args=None):
    """
    Each scenario is paired by furthest distance
        - bun_size (approx. size of each bundle) can be passed into bundle_args. default is 2.
        - ensure there are no duplicate/redundant scenarios before using this! 
        - all scenarios must have unique names! 
    """
    if models is None:
        models = list(data.keys())

    if "bun_size" in bundle_args:    # default bundle size is 2
        bun_size = bundle_args['bun_size']
    else:
        bun_size = 2

    num_scens = sum(len(data[model]) for model in models)   # total number of scenarios
    if bun_size > num_scens:
        raise ValueError(f"Bundle size cannot exceed number of scenarios")

    num_centers = -(num_scens// -bun_size)  # number of bundle centers (NOT NECESSARILY the same as number of bundles!!)
    all_scens = {sname: sval for model in models for (sname, sval) in data[model].items()}

    if isinstance(data[models[0]][next(iter(all_scens))]['Demand'], numbers.Number) == True:
        arr = np.array([all_scens[s]['Demand'] for s in all_scens.keys()])
        X = arr.reshape(-1,1)   # array of scenario demands needs to be reshaped if demand is a number
    else:
        X = np.array([all_scens[s]['Demand'] for s in all_scens.keys()])

    kmeans = KMeans(n_clusters=num_centers, random_state=0, n_init="auto").fit(X)   # find bundle centers
    centers = kmeans.cluster_centers_  # list of bundle centers

    max_diffs = {}
    for s in all_scens:    # map scenarios to furthest bundle center
        diffs = [float(np.linalg.norm(centers[i] - all_scens[s]['Demand'])) for i in range(len(centers))]
        max_diffs[s] = diffs.index(max(diffs))

    bundle = {}
    for model in models:
        for s in data[model]:
            if f"bundle_{max_diffs[s]}" in bundle:  # ensures empty bundles aren't created if centers have no mapped scenarios
                bundle[f"bundle_{max_diffs[s]}"]["scenarios"].update({scen_key(model, s): data[model][s]['Probability']})
            else:
                bundle[f"bundle_{max_diffs[s]}"] = dict(
                    scenarios = {scen_key(model, s): data[model][s]['Probability']},
                    Probability = 1/num_centers
                )

    # normalization term for if some centers have no mapped scenarios
    fewer_centers_norm = sum(bundle[bkey]["Probability"] for bkey in bundle.keys())
    for bkey in bundle.keys():
        bundle[bkey]["Probability"] *= (1/fewer_centers_norm)
        norm_factor = sum(bundle[bkey]["scenarios"].values())   # normalizing bundle probabilities
        for skey in bundle[bkey]["scenarios"].keys():
            bundle[bkey]["scenarios"][skey] *= (1/norm_factor)

    return bundle


def bundle_random_partition(data, model_weight, models=None, bundle_args=None):
    """
    Each scenario is randomly assigned to a single bundle
        - Need to pass number of bundles (num_buns) to bundle_args
    """
    if models is None:
        models = list(data.keys())

    if "num_buns" not in bundle_args:
        raise RuntimeError(f"Need to include the number of bundles (num_buns) in bundle_args")

    # user can optionally set random seed
    seed_value = 972819128347298
    if bundle_args != None:
        seed_value = bundle_args.get("seed", seed_value)
        num_buns = bundle_args["num_buns"]
    random.seed(seed_value)

    scens = [data[fid] for fid in data.keys()]
    #scens = {sname: sval for model in models for (sname, sval) in data[model].items()}

    # extracting number of bundles from bundle_args
    num_buns = bundle_args["num_buns"]
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
    for idx, _ in enumerate(
        scens
    ):  # using indices rather than IDs to allow for non-numeric scenario IDs
        scen_idx.append(idx)

    temp_bundle = []  # randomly assign scenarios to bundles
    for bun_idx in range(num_buns):
        temp_list = random.sample(scen_idx, bunsize[bun_idx])
        temp_bundle.append(temp_list)
        for temp_scen_idx in temp_list:
            scen_idx.remove(temp_scen_idx)
    print(scens[temp_bundle[0][0]])
    if len(scen_idx) != 0:  # check that each scenario is assigned to a bundle
        raise RuntimeError(f"Scenarios {scen_idx} are not assigned to a bundle")

    bundle = {}
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
    "single_scenario": single_scenario,
    "single_bundle": single_bundle,
    "bundle_random_partition": bundle_random_partition,
    "mf_paired": mf_paired,
    "mf_random_nested": mf_random_nested,
    "mf_random": mf_random,
    "mf_ordered": mf_ordered,
    "similar_partitions": similar_partitions,
    "dissimilar_partitions": dissimilar_partitions,
    "kmeans_similar": kmeans_similar, 
    "kmeans_dissimilar": kmeans_dissimilar
}


def bundle_scheme(data, scheme_str, model_weight, models, bundle_args=None):
    bundle = scheme[scheme_str](data, model_weight, models, bundle_args)

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

    def __init__(
        self,
        *,
        data=None,
        scheme=None,
        models=None,
        model_weight=None,
        bundle_args=None,
    ):
        if scheme == None:
            # Empty constructor
            return

        self.bundle_scheme_str = scheme
        self.bundle_models = models
        self.bundle_weights = model_weight
        self.bundle_args = bundle_args
        bundles = bundle_scheme(data, scheme, model_weight, models, bundle_args)
        self._bundles = {
            key: munch.Munch(
                probability=bundles[key]["Probability"],
                scenarios=list(sorted(bundles[key]["scenarios"].keys())),
                scenario_probability=bundles[key]["scenarios"],
            )
            for key in bundles
        }

    def to_dict(self):
        return munch.unmunchify(self._bundles)

    def __len__(self):
        return len(self._bundles)

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

    def dump(self, json_filename, indent=None, sort_keys=False):
        data = dict(
            scheme=self.bundle_scheme_str,
            models=self.bundle_models,
            weights=self.bundle_weights,
            args=self.bundle_args,
            bundles=self.to_dict(),
        )
        with open(json_filename, "w") as OUTPUT:
            json.dump(JSencoded(data), OUTPUT, indent=indent, sort_keys=sort_keys)


def create_bundles(data):
    # TODO: error checking on data fields
    bundles = BundleObj()
    bundles.bundle_scheme_str = data["scheme"]
    bundles.bundle_models = data["models"]
    bundles.bundle_weights = data["weights"]
    bundles.bundle_args = data["args"]
    bundles._bundles = data["bundles"]
    return bundles


def load_bundles(filename):
    with open(filename, "r") as INPUT:
        data = json.load(INPUT, cls=JSdecoded)
    return create_bundles(data)
