import ast
import json
import munch
import types

from .SF_schemes import *
from .MF_schemes import *


def _JSdecoded(item: dict, dict_key=False):
    """
    Decode JSON data to Python objects.

    Parameters
    ----------
    item : dict
        The item to decode.
    dict_key : bool, optional
        Whether the item is a dictionary key (default is False).

    Returns
    -------
    object
        The decoded item.
    """
    if isinstance(item, list):
        return [JSdecoded(e) for e in item]
    elif isinstance(item, dict):
        return {ast.literal_eval(key): value for key, value in item.items()}
    return item


def _JSencoded(item, dict_key=False):
    """
    Encode Python objects to JSON-compatible data.

    Parameters
    ----------
    item : object
        The item to encode.
    dict_key : bool, optional
        Whether the item is a dictionary key (default is False).

    Returns
    -------
    object
        The encoded item.
    """
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
    "mf_similar_partitions": mf_similar_partitions,
    "mf_dissimilar_partitions": mf_dissimilar_partitions,
    "kmeans_similar": kmeans_similar,
    "kmeans_dissimilar": kmeans_dissimilar,
    "mf_kmeans_dissimilar": mf_kmeans_dissimilar,
    "mf_kmeans_similar": mf_kmeans_similar,
    "mf_bundle_from_list": mf_bundle_from_list,
    "bundle_from_list": bundle_from_list,
}


def _is_multifidelity(scheme_str):
    if scheme_str == None:
        scheme_str = "single_scenario"
    if scheme_str[:3] == "mf_":
        return True
    else:
        return False

def _bundle_scheme(data, scheme_str, models, model_weight=None, bundle_args=None):
    """
    Create bundles using a specified bundling scheme.

    Parameters
    ----------
    data : dict
        The data to bundle.
    scheme_str : str
        The bundling scheme to use.
    models : list
        List of model names.
    model_weight : dict, optional
        Dictionary of model weights.
    bundle_args : dict, optional
        Additional arguments for the bundling scheme.

    Returns
    -------
    dict
        The created bundles.

    Raises
    ------
    RuntimeError
        If bundle probabilities or scenario probabilities do not sum to 1.
    """
    if model_weight:
        bundle = scheme[scheme_str](data, model_weight, models, bundle_args)
    else:
        bundle = scheme[scheme_str](data, models, bundle_args)

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
    """
    A class to represent bundles of scenarios.

    Attributes
    ----------
    bundle_scheme_str : str or None
        The bundling scheme used.
    bundle_models : list or None
        List of model names.
    bundle_weights : dict or None
        Dictionary of model weights.
    bundle_args : dict or None
        Additional arguments for the bundling scheme.
    _bundles : dict
        Dictionary of bundles.
    """

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
            bundles = _bundle_scheme(data, scheme, model_weight, models, bundle_args)
        else:
            bundles = _bundle_scheme(data, scheme, models, bundle_args)

        self._bundles = {
            key: munch.Munch(
                probability=bundles[key]["Probability"],
                scenarios=list(sorted(bundles[key]["scenarios"].keys())),
                scenario_probability=bundles[key]["scenarios"],
            )
            for key in bundles
        }

    def to_dict(self):
        """
        Convert the bundles to a dictionary.

        Returns
        -------
        dict
            The bundles as a dictionary.
        """
        return munch.unmunchify(self._bundles)

    def __len__(self):
        """
        Get the number of bundles.

        Returns
        -------
        int
            The number of bundles.
        """
        return len(self._bundles)

    def __contains__(self, key):
        """
        Check if a key is in the bundles.

        Parameters
        ----------
        key : object
            The key to check.

        Returns
        -------
        bool
            True if the key is in the bundles, False otherwise.
        """
        return key in self._bundles

    def __getitem__(self, key):
        """
        Get a bundle by key.

        Parameters
        ----------
        key : object
            The key of the bundle to get.

        Returns
        -------
        object
            The bundle corresponding to the key.

        Raises
        ------
        AssertionError
            If the key is not in the bundles.
        """
        assert (
            key in self._bundles
        ), f"Unexpected key {key} {type(key)}.  Valid keys: {list(self._bundles.keys())}"
        return self._bundles[key]

    def __iter__(self):
        """
        Iterate over the bundle keys.

        Yields
        ------
        object
            The keys of the bundles.
        """
        for key in self._bundles:
            yield key

    def keys(self):
        """
        Get the keys of the bundles.

        Yields
        ------
        object
            The keys of the bundles.
        """
        for key in self._bundles:
            yield key

    def dump(self, json_filename, indent=None, sort_keys=False):
        """
        Dump the bundles to a JSON file.

        Parameters
        ----------
        json_filename : str
            Path to the JSON file.
        indent : int, optional
            Indentation level for JSON formatting.
        sort_keys : bool, optional
            Whether to sort keys in the JSON output.
        """
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
    """
    Create bundles from data.

    Parameters
    ----------
    data : dict
        Dictionary containing bundle data.

    Returns
    -------
    BundleObj
        The created BundleObj instance.
    """
    # TODO: error checking on data fields
    bundles = BundleObj()
    bundles.bundle_scheme_str = data["scheme"]
    bundles.bundle_models = data["models"]
    bundles.bundle_weights = data["weights"]
    bundles.bundle_args = data["args"]
    bundles._bundles = data["bundles"]
    return bundles


def load_bundles(filename):
    """
    Load bundles from a JSON file.

    Parameters
    ----------
    filename : str
        Path to the JSON file.

    Returns
    -------
    BundleObj
        The loaded BundleObj instance.
    """
    with open(filename, "r") as INPUT:
        data = json.load(INPUT, cls=_JSdecoded)
    return create_bundles(data)
