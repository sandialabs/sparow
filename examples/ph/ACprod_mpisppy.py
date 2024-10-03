# The AC production planning example for general agnostic with Pyomo as guest language
# This example includes bundles as an option
# ALL INDEXES ARE ZERO-BASED
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2018 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
from mpisppy.utils import config
import json
with open('mpisppy/agnostic/scendata.json', 'r') as file:
    data = json.load(file)

'''
    IMPORTANT:
        - IF USING BUNDLES: fidelity is "bun"
        - IF NOT USING BUNDLES: specify fidelity (e.g., "HF", "LF").
                                if more than two fidelities (HF,LF), edit scenario_names_creator
'''

fidelity = "LF" # should be "HF" or "LF" if using scenarios; "bun" if using bundles

# load in the number of HF and LF scenarios
num_HFscens = len(data['scenarios']['HF'])
num_LFscens = len(data['scenarios']['LF'])

if fidelity == "bun":
    numbuns = len(data['bundles']['MF'])
    bunsize = len(data['bundles']['MF'][0]['HF']) + len(data['bundles']['MF'][0]['LF']) # assuming all bundles are the same size!
    original_num_scens = num_HFscens + num_LFscens
else:
    numbuns = 0
    bunsize = 0
    original_num_scens = None

# lists that map scenario/bundle ID to order in json list
if fidelity == "HF":
    scenID_HFlist = []
    for i in range(num_HFscens):
        scenID_HFlist.append(data['scenarios']['HF'][i]['ID'])
else:
    scenID_LFlist = []
    for j in range(num_LFscens):
        scenID_LFlist.append(data['scenarios']['LF'][j]['ID'])

if numbuns > 0:
    bundID_list = []
    for i in range(numbuns):
        bundID_list.append(data['bundles']['MF'][i]['ID'])

def scenario_creator(scenario_name, use_integer=False, sense=pyo.minimize, num_scens=None):
    """ Create a scenario for the AC production planning example.
    
    Args:
        scenario_name (str):
            Name of the scenario to construct, which might be a bundle.
        use_integer (bool, optional):
            If True, restricts variables to be integer. Default is False.
        sense (int, optional):
            Model sense (minimization or maximization). Must be either
            pyo.minimize or pyo.maximize. Default is pyo.minimize.
        num_scens (int, optional):
            Number of scenarios. We use it to compute _mpisppy_probability. 
            Default is None.
    """
    if "HF" == scenario_name[:2] or "LF" == scenario_name[:2]:
        # scenario_name has the form <str>_<int> e.g. HF_12, LF_7
        scen_fidelity = scenario_name[:2]               # either HF or LF
        scennum       = int(float(scenario_name[3:]))   # matches ID associated with scenario
        scenname      = scen_fidelity+'_'+str(scennum)

        # Check for minimization vs. maximization
        if sense not in [pyo.minimize, pyo.maximize]:
            raise ValueError("Model sense Not recognized")

        # Create the concrete model object
        model = pysp_instance_creation_callback(
            scenname,
            use_integer=use_integer,
            sense=sense
        )

        # create a varlist, which is used to create a vardata list
        # (This list needs to whatever the guest needs, not what Pyomo needs)
        varlist = [model.RegularTime[1], model.Overtime[1], model.Storage[1]]
        model._nonant_vardata_list = sputils.build_vardatalist(model, varlist)
        sputils.attach_root_node(model, model.FirstStageCost, varlist)

        # Add the probability of the scenario -- assumed to be uniform unless specified otherwise
        if num_scens is not None :
            if scen_fidelity == "HF":
                model._mpisppy_probability = data['scenarios'][scen_fidelity][scenID_HFlist.index(scennum)]["Probability"]
            elif scen_fidelity == "LF":
                model._mpisppy_probability = data['scenarios'][scen_fidelity][scenID_LFlist.index(scennum)]["Probability"]
            else:
                raise RuntimeError (f"No scenario fidelity specified")
        else:
            model._mpisppy_probability = "uniform"
        return model
        
    elif "bund" == scenario_name[:4] or "Bund" == scenario_name[:4]:
        bundnum = int(float(scenario_name[7:]))     # matches ID associated with bundle
        '''
        bundles are the same size... for now:
        '''
        assert len(data['bundles']['MF'][bundID_list.index(bundnum)]['LF']) + len(data['bundles']['MF'][bundID_list.index(bundnum)]['HF']) == bunsize

        # name each LF and HF scenario in the bundle
        snames = [f"HF_{data['bundles']['MF'][bundID_list.index(bundnum)]['HF'][i]}" for i in range(len(data['bundles']['MF'][bundID_list.index(bundnum)]['HF']))]
        snames.extend([f"LF_{data['bundles']['MF'][bundID_list.index(bundnum)]['LF'][j]}" for j in range(len(data['bundles']['MF'][bundID_list.index(bundnum)]['LF']))])

        bunkwargs = {"use_integer": use_integer,
                     "sense": sense, 
                     "num_scens":None}

        # call scenario_creator with snames:
        bundle = sputils.create_EF(snames, scenario_creator,
                                   scenario_creator_kwargs=bunkwargs,
                                   EF_name=scenario_name,
                                   nonant_for_fixed_vars = False)
        # It simplifies things if we assume that it is a 2-stage problem,
        # or that the bundles consume entire second stage nodes,
        # then all we need is a root node and the only nonants that need to be reported are
        # at the root node (otherwise, more coding is required here to figure out which nodes and Vars
        # are shared with other bundles)
        nonantlist = [v for idx,v in bundle.ref_vars.items() if idx[0] =="ROOT"]
        sputils.attach_root_node(bundle, 0, nonantlist)
        # scenarios are equally likely so bundles are too
        bundle._mpisppy_probability = data['bundles']['MF'][bundID_list.index(bundnum)]["Probability"]
        return bundle
    else:
        raise RuntimeError (f"Scenario name does not have scen or bund: {snames}")

def pysp_instance_creation_callback(
    scenario_name, use_integer=False, sense=pyo.minimize):
    # long function to create the entire model
    # scenario_name is a string (e.g. LF_0)
    # Returns a concrete model for the specified scenario
    scennum = int(float(scenario_name.strip("FHL_")))
    scen_fidelity = str(scenario_name.rstrip("0123456789_"))

    model = pyo.ConcreteModel(scenario_name)

    ## DEMAND PARAMS
    Demand = {}
    Demand['HF'] = {data['scenarios']['HF'][i]['ID']: data['scenarios']['HF'][i]['Demand'] for i in range(len(data['scenarios']['HF']))}
    Demand['LF'] = {data['scenarios']['LF'][j]['ID']: data['scenarios']['LF'][j]['Demand'] for j in range(len(data['scenarios']['LF']))}

    def Demand_init(m):
        return Demand[scen_fidelity][scennum]
    
    model.Demands = pyo.Param(within=pyo.NonNegativeReals, initialize=Demand_init, mutable=True)

    ## SETS

    model.Time = pyo.Set(initialize=[1,2]) # number of time periods (months)

    ## VARIABLES

    model.RegularTime = pyo.Var(model.Time, within=pyo.NonNegativeIntegers) # number of units produced in reg. time at month t
    model.Storage     = pyo.Var(model.Time, within=pyo.NonNegativeIntegers) # number of units stored at month t
    model.Overtime    = pyo.Var(model.Time, within=pyo.NonNegativeIntegers) # number of units produced in OT at month t

    ## CONSTRAINTS

    def RTProduction_rule(model, t): # max. two units produced in reg. time for each month
        return model.RegularTime[t] <= 2
    model.RTProduction = pyo.Constraint(model.Time, rule=RTProduction_rule)

    def Month1Demand_rule(model): # total demand is 1 for month 1
        return model.RegularTime[1] + model.Overtime[1] - model.Storage[1] == 1
    model.Month1Demand = pyo.Constraint(rule=Month1Demand_rule)

    def Month2Demand_rule(model): # total demand in month 2 is uncertain
        return model.Storage[1] + model.RegularTime[2] + model.Overtime[2] - model.Storage[2] == model.Demands
    model.Month2Demand = pyo.Constraint(rule=Month2Demand_rule)

    ## OBJECTIVE

    def ComputeFirstStageCost_rule(model): # production costs for month 1
        return model.RegularTime[1] + 3*model.Overtime[1] + 0.5*model.Storage[1]
    model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

    def ComputeSecondStageCost_rule(model): # production costs for month 2
        return model.RegularTime[2] + 3*model.Overtime[2] + 0.5*model.Storage[2]
    model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

    def total_cost_rule(model): # total production costs over time horizon
        return model.FirstStageCost + model.SecondStageCost
    model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, sense=sense)


    return model


# begin helper functions
#=========
def scenario_names_creator(num_scens, start=None):
    # return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    #print(f"names_creator {bunsize=}")
    if (start is None) :
        start=0
    if bunsize == 0:
        # return an array of scenario names, starting with HF and ending with LF
        if fidelity == "HF":
            scenvec = [f"HF_{data['scenarios']['HF'][i]['ID']}" for i in range(start,start+num_HFscens)]
        elif fidelity == "LF":
            scenvec = [f"LF_{data['scenarios']['LF'][j]['ID']}" for j in range(start,start+num_LFscens)]
        else:
            raise RuntimeError (f"No scenario fidelity specified")
        return scenvec
    else:
        # The hack should have changed the value of num_scens to be a fib!
        # We will assume that start and and num_scens refers to bundle counts.
        # Bundle numbers are zero based and scenario numbers as well.
        return [f"bundle_{data['bundles']['MF'][i]['ID']}" for i in range(start,len(data['bundles']['MF']))]
    

#=========
def inparser_adder(cfg):
    cfg.num_scens_required()
    
    cfg.add_to_config("farmer_with_integers",
                      description="make the version that has integers (default False)",
                      domain=bool,
                      default=False)
    cfg.add_to_config("bundle_size",
                      description="number of scenarios per bundle (default 0, which means no bundles, as does 1)",
                      domain=int,
                      default=0)
    


#=========
def kw_creator(cfg):
    # (for Amalgamator): linked to the scenario_creator and inparser_adder
    kwargs = {"use_integer": cfg.get('farmer_with_integers', False),
              "num_scens" : cfg.get('num_scens', None),
              }
    return kwargs

def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    """ Create a scenario within a sample tree. Mainly for multi-stage and simple for two-stage.
        (this function supports zhat and confidence interval code)
    Args:
        sname (string): scenario name to be created
        stage (int >=1 ): for stages > 1, fix data based on sname in earlier stages
        sample_branching_factors (list of ints): branching factors for the sample tree
        seed (int): To allow random sampling (for some problems, it might be scenario offset)
        given_scenario (Pyomo concrete model): if not None, use this to get data for ealier stages
        scenario_creator_kwargs (dict): keyword args for the standard scenario creator funcion
    Returns:
        scenario (Pyomo concrete model): A scenario for sname with data in stages < stage determined
                                         by the arguments
    """
    sca = scenario_creator_kwargs.copy()
    sca["num_scens"] = sample_branching_factors[0]  # two-stage problem
    return scenario_creator(sname, **sca)


# end helper functions


#============================
def scenario_denouement(rank, scenario_name, scenario):
    sname = scenario_name
    #print("denouement needs work")
    #scenario.pprint()
    return
    s = scenario
    if sname == 'scen0':
        print("Arbitrary sanity checks:")
       # print ("SUGAR_BEETS0 for scenario",sname,"is",
       #        pyo.value(s.DevotedAcreage["SUGAR_BEETS0"]))
        print ("FirstStageCost for scenario",sname,"is", pyo.value(s.FirstStageCost))


# special helper function (hack) for bundles
def bundle_hack(cfg):
    # Hack to put bundle information in global variables to be used by
    # the names creator.  (only relevant for bundles)
    # numbuns and bunsize are globals with default value 0
    if cfg.bundle_size > 1:
        assert cfg.num_scens % cfg.bundle_size == 0,\
            "Due to laziness, the bundle size must divide the number of scenarios"
        global bunsize, numbuns, original_num_scens
        bunsize = cfg.bundle_size
        numbuns = cfg.num_scens // cfg.bundle_size
        original_num_scens = cfg.num_scens
        cfg.num_scens = numbuns

