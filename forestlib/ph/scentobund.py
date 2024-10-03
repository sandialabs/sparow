'''
bundle is a dictionary of dictionaries
    - keys are names of bundles
    - for each dictionary in bundle, keys are 'IDs' (i.e., which scenarios are in the bundle) and 'Probability'

specify which bundling scheme (function) is used via "bundle_scheme" in sp.py
'''

scheme = {'bundle_by_fidelity':       bundle_by_fidelity,
          'bundle_multifid':          bundle_multifid, 
          'bundle_similar_partition': bundle_similar_partition,
          'single_scenario':          single_scenario}

def bundle_scheme(data, scheme_str, bundle_args=None):
        return scheme[scheme_str](data)

###################################################################################################################

bundle = {}

def bundle_by_fidelity(data):

    bundle_names = ['HF', 'LF']
    for bund in bundle_names:
        bundle[bund] = {}

    scenID_HFlist = []
    scenID_LFlist = []
    for i in range(len(data['scenarios'])):
        if data['scenarios'][i]['Fidelity'] == 'HF':
            scenID_HFlist.append(data['scenarios'][i]['ID'])
        elif data['scenarios'][i]['Fidelity'] == 'LF':
            scenID_LFlist.append(data['scenarios'][i]['ID'])
        else:
            raise RuntimeError (f"No fidelity specified for {i}th scenario")
    # will sum to 1 b/c uniform distribution
    bundle['HF'] = {'IDs':         scenID_HFlist,
                    'Probability': sum(data['scenarios'][scenID_HFlist.index(j)]['Probability'] for j in scenID_HFlist)}
    bundle['LF'] = {'IDs':         scenID_LFlist,
                    'Probability': sum(data['scenarios'][scenID_LFlist.index(j)]['Probability'] for j in scenID_LFlist)}

    return bundle


def bundle_multifid(data):  # still needs some work
    bundle_names = ['Low', 'Medium', 'High']

    for bund in bundle_names:
        bundle[bund] = {}

    bundle['Low']    = {'IDs': [0, 1, 6],    'Probability': 0.2}
    bundle['Medium'] = {'IDs': [2, 3, 7, 8], 'Probability': 0.5}
    bundle['High']   = {'IDs': [4, 5, 9],    'Probability': 0.3}

    return bundle


def bundle_similar_partition(data): # bundle similar scenarios together; each scenario appears in exactly one bundle
    '''
    DETERMINE NUMBER OF BUNDLES
    '''
    # lower/upper bounds on the number of bundles
    min_num_buns = 1
    max_num_buns = 3    # difference between min and max should be at least 1

    list_scens = data['scenarios']

    list_buns = {0: [[[0,1,2]]], 
                 1: [[[0,1], [2]], [[0,2], [1]], [[1,2], [0]]], 
                 2: [[[0]], [[1]], [[2]]]} ### need to think about how to generalize this
    bun_err = []
    for i in range(max_num_buns - min_num_buns + 1):
        bun_mean = []                       # initialize list of bundle means when the number of bundles is i
        temp_bun_err = 0                    # initialize distance from each scenario to its assigned bundle when the number of bundles is i
        for j in range(len(list_buns[i])):  # assign each scenario to closest bundle:
            # compute mean of each bundle
            bun_mean.append([(sum(list_buns[i][j][m])/len(list_buns[i][j][m])) for m in range(len(list_buns[i][j]))])
            temp_dict = {m: [] for m in range(len(list_buns[i][j]))} # store each scenario's assignment to a bundle
            
            ### need to add if statement to eliminate redundant bundle means / skip bundle reassignment if all bundle means are the same #if all(q == bun_mean[0] for q in bun_mean): 
            for k, scen in enumerate(list_scens): # scenario-bundle assignment based on min distance to bundle mean
                dist_scen_to_bun = [abs(scen['Demand'] - bun_mean[j][m]) for m in range(len(list_buns[i][j]))]
                min_dist = min(dist_scen_to_bun)
                temp_dict[dist_scen_to_bun.index(min_dist)].append(scen)
                temp_bun_err += (abs(bun_mean[j][dist_scen_to_bun.index(min_dist)] - scen['Demand']))

            # calculate bundling error 
        bun_err.append(temp_bun_err)
    
    total_bun_err = [abs(bun_err[i] - bun_err[i-1]) for i in range(1, max_num_buns - min_num_buns + 1)]
    max_total_bun_err = max(total_bun_err)
    num_buns = total_bun_err.index(max_total_bun_err) + min_num_buns + 1

    '''
    CREATE num_buns BUNDLES
    '''
    core_scenarios = {} # core scenarios are LF (can change depending on the problem)
    for i in range(len(data['scenarios'])):
        if data['scenarios'][i]['Fidelity'] == 'LF':
            core_scenarios.update({f'{data['scenarios'][i]['ID']}': data['scenarios'][i]['Demand']})

    # assert len(core_scenarios) == num_buns

    for s in core_scenarios.keys():
        bundle[s] = {'IDs': [], 'Probability': []}

    not_core_scenarios = {}
    for i in range(len(data['scenarios'])):
        if data['scenarios'][i]['Fidelity'] == 'HF':
            not_core_scenarios.update({f'{data['scenarios'][i]['ID']}': [data['scenarios'][i]['Demand'], data['scenarios'][i]['Probability']]})

    dist_to_cores = []
    for k, scen in enumerate(not_core_scenarios):
        for s, core in enumerate(core_scenarios.keys()):
            dist_to_cores.append(abs(not_core_scenarios[scen][0] - core_scenarios[core]))
            
        min_dist = min(dist_to_cores)
        closest_core = list(core_scenarios.items())[dist_to_cores.index(min_dist)][0]
        bundle[f'{closest_core}']['IDs'].append(scen)
        bundle[f'{closest_core}']['Probability'].append(not_core_scenarios[scen][1])
        dist_to_cores = []  # reset dist_to_cores
        
    # eliminate cores that have no associated scenarios; append to nearest core 
    uncore_scens = {}
    for n, key in enumerate(core_scenarios):
        if len(bundle[key]['IDs']) == 0:
            uncore_scens.update({key: data['scenarios'][n]})
            del bundle[key]

    for key in uncore_scens:
        del core_scenarios[key]

    if len(uncore_scens) == 0:
        pass
    else:
        for uc_scen in uncore_scens:
            for s, core in enumerate(core_scenarios.keys()):
                dist_to_cores.append(abs(uncore_scens[uc_scen]['Demand'] - core_scenarios[core]))
            
            min_dist = min(dist_to_cores)
            closest_core = list(core_scenarios.items())[dist_to_cores.index(min_dist)][0]
            bundle[f'{closest_core}']['IDs'].append(uc_scen)
            bundle[f'{closest_core}']['Probability'].append(uncore_scens[uc_scen]['Probability'])
            dist_to_cores = []  # reset dist_to_cores

    return bundle 


def single_scenario(data): # only using HF scenarios!!
    for scen in data['scenarios']:
        bundle[str(scen['ID'])] = {'IDs': [scen['ID']], 'Probability': scen['Probability']}

    return bundle


def bundle_similar_cover(data): # bundle similar scenarios together; each scenario appears in two bundles
    pass 


def bundle_random_partition(data): # random bundling
    pass

