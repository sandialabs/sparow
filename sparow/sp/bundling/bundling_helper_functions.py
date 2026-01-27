import warnings

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


def scen_name(model, scenario):
    if model is None:
        return f"{scenario}"
    return f"{model}_{scenario}"


def scen_key(model, scenario):
    return (model, scenario)


def check_data_dict_keys(data, model0, bundle_args, dkey_required=False):
    # by default, specifying a demand key is not required
    dkey = None  # initialize demand and probability keys
    pkey = None

    if bundle_args:  # will populate dkey and pkey if they're provided
        dkey = bundle_args.get("demand_key")
        pkey = bundle_args.get("probability_key")

    if dkey is None and dkey_required == True:
        # search first item in HF data dictionary for demand key
        dkeys_to_check = ["Demand", "demand", "DEMAND", "d", "D"]
        existing_dkey = [
            e_dkey
            for e_dkey in dkeys_to_check
            if e_dkey in data[model0][next(iter(data[model0]))].keys()
        ]
        if (len(existing_dkey) > 1 or len(existing_dkey) == 0):
            raise RuntimeError(f"Specify demand_key in bundle_args")
        dkey = existing_dkey[0]
    
    if pkey is None:
        # search entire data dictionary for probability key(s)
        pkeys_to_check = [
            "Probability",
            "probability",
            "PROBABILITY",
            "prob",
            "Prob",
            "p",
            "P",
            "Pr",
            "pr",
        ]
        all_scens = list(data.values())
        list_all_scens = [list(all_scens[s].values()) for s in range(len(all_scens))]
        all_keys = [
            list(item.keys())
            for l in range(len(list_all_scens))
            for item in list_all_scens[l]
        ]
        flat_list = list(
            set([list_item for sublist in all_keys for list_item in sublist])
        )
        existing_pkey = [list_item for list_item in flat_list if list_item in pkeys_to_check]
        if len(existing_pkey) > 1:
            raise RuntimeError(f"Specify probability_key in bundle_args")
        elif len(existing_pkey) == 0:
            pkey = None
            warnings.warn(
                "No scenario probabilities are given; assuming uniform distribution.",
                UserWarning,
            )
        else:
            pkey = existing_pkey[0]

    return dkey, pkey
