from typing import Any, List
from . import bundling_functions


def initialize_bundles(
    *,
    scheme: str | None = None,
    models: list[str] | None = None,
    default_model: str | None = None,
    model_data: dict[str, Any] | None = None,
    scenario_data: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Initialize bundles for stochastic programming.

    Parameters
    ----------
    scheme : str, optional
        The bundling scheme to use. Defaults to "single_scenario".
    models : list, optional
        List of model names to include in the bundles.
    default_model : str, optional
        The default model name.
    model_data : dict, optional
        Dictionary containing model data.
    scenario_data : dict, optional
        Dictionary containing scenario data.
    **kwargs : dict
        Additional keyword arguments for bundling.

    Returns
    -------
    BundleObj
        An initialized BundleObj instance.
    """
    if scenario_data is None:
        scenario_data = {}
    if model_data is None:
        model_data = {}

    if scheme == None:
        scheme = "single_scenario"
    if models == None:
        models = [default_model] + list(
            sorted(model for model in scenario_data.keys() if model != default_model)
        )
    else:
        for name in models:
            assert name in scenario_data

    assert len(models) > 0, "Cannot initialize bundles without model data"
    if "model_weight" in kwargs:
        model_weight = kwargs["model_weight"]
    else:
        model_weight = {
            model: mdata.get("_model_weight_", 1.0)
            for model, mdata in model_data.items()
        }

    if model_weight:
        return bundling_functions.BundleObj(
            data=scenario_data,
            models=models,
            model_weight=model_weight,
            scheme=scheme,
            bundle_args=kwargs,
        )
    else:
        return bundling_functions.BundleObj(
            data=scenario_data,
            models=models,
            scheme=scheme,
            bundle_args=kwargs,
        )
