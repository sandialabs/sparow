import ast
import json
import munch
import types

from .SF_schemes import *
from .MF_schemes import *

# import MF_schemes
# import SF_schemes


def JSdecoded(item: dict, dict_key=False):
    if isinstance(item, list):
        return [JSdecoded(e) for e in item]
    elif isinstance(item, dict):
        return {ast.literal_eval(key): value for key, value in item.items()}
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


scheme = {
    "single_scenario": single_scenario,
    "single_bundle": single_bundle,
    "sf_random": sf_random,
    "mf_paired": mf_paired,
    "mf_random_nested": mf_random_nested,
    "mf_random": mf_random,
    "similar_partitions": similar_partitions,
    "dissimilar_partitions": dissimilar_partitions,
    "kmeans_similar": kmeans_similar,
    "kmeans_dissimilar": kmeans_dissimilar,
    "mf_kmeans_dissimilar": mf_kmeans_dissimilar,
    "mf_kmeans_similar": mf_kmeans_similar,
    "mf_bundle_from_list": mf_bundle_from_list,
    "bundle_from_list": bundle_from_list,
}


def bundle_scheme(data, scheme_str, models, model_weight=None, bundle_args=None):
    if model_weight:
        bundle = scheme[scheme_str](data, model_weight, models, bundle_args)
    else:
        bundle = scheme[scheme_str](data, models, bundle_args)

    # model0 = models[0]
    pkey = "Probability"

    # Return error if bundle probabilities do not sum to 1
    if abs(sum(b[pkey] for b in bundle.values()) - 1.0) > 1e-04:
        raise RuntimeError(
            f"Bundle probabilities sum to {sum(bundle[key][pkey] for key in bundle)}"
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

        if model_weight:
            bundles = bundle_scheme(data, scheme, model_weight, models, bundle_args)
        else:
            bundles = bundle_scheme(data, scheme, models, bundle_args)
        # model0 = models[0]
        # pkey = check_data_dict_keys(data, model0, bundle_args)[1]
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
