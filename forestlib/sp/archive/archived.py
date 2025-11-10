"""
Code that's been archived
"""

### ARCHIVED FROM sp_pyomo.py
""" class StochasticProgram_Pyomo_MultistageBuilder(StochasticProgram_Pyomo_Base):

    def __init__(self, *, model_builder_list):
        super().__init__()
        assert (type(model_builder_list) is list) and (
            len(model_builder_list) >= 2
        ), "Expecting a list of model_builder functions with length >= 2"
        assert (
            len(model_builder_list) == 2
        ), "WEH - This class only works for two-stages right now."
        self.model_builder_list = model_builder_list

    def initialize_model(self, *, name=None, filename=None, model_data=None, **kwargs):
        if filename is not None:
            with open(f"{filename}", "r") as file:
                model_data = json.load(filename)

        if model_data is not None:
            self.model_data[name] = model_data.get("data", {})
            self.scenario_data[name] = {
                scen["ID"]: scen for scen in model_data.get("scenarios", {})
            }

        if model_data is not None:
            self.initialize_bundles(models=[name])

    def _first_stage_variables(self, *, M):
        for var in find_variables(M):
            yield var.name, var

    def _create_scenario(self, M, s):
        model_name, scenario = s
        data = copy.copy(self.app_data)
        for k, v in self.model_data.get(model_name, {}).items():
            assert k not in data, f"Model data for {k} has already been specified!"
            data[k] = v
        for k, v in self.scenario_data[model_name].get(scenario, {}).items():
            assert k not in data, f"Scenario data for {k} has already been specified!"
            data[k] = v
        self.model_builder_list[1](M, M.s[s], data, {})

    def create_EF(self, *, w=None, x_bar=None, rho=None, b, cached=False):
        scenarios = self.bundles[b].scenarios

        # 1) create EF model
        EF_model = pyo.ConcreteModel()
        self.model_builder_list[0](EF_model, self.app_data, {})
        # Find the root objective
        root_obj = find_objective(EF_model)
        # Initialize the cuid_map, and also initialize the int_to_FirstStageVar map for this bundle
        self._initialize_cuid_map(M=EF_model, b=b)

        # 2) Loop through scenario dictionary, add block, deactivate Obj
        EF_model.s = pyo.Block(scenarios)
        obj_comp = {}
        for s in scenarios:
            self._create_scenario(EF_model, s)
            obj_comp[s] = find_objective(EF_model.s[s])
            assert (
                obj_comp[s] is not None
            ), f"Cannot find objective on block for scenario '{s}'"
            obj_comp[s].deactivate()

        # 3)Create Obj: root_obj + (sum of scenario obj * probability)
        obj = 0 if root_obj is None else root_obj
        obj = obj + sum(
            self.bundles[b].scenario_probability[s] * obj_comp[s].expr
            for s in scenarios
        )
        if w is not None:
            obj = (
                obj
                + sum(w[i] * x for i, x in self.int_to_FirstStageVar[b].items())
                + sum(
                    (rho[i] / 2.0) * ((x - x_bar[i]) ** 2)
                    for i, x in self.int_to_FirstStageVar[b].items()
                )
            )
        EF_model.obj = pyo.Objective(expr=obj)
        if root_obj is not None:
            root_obj.deactivate()

        return EF_model
 """
