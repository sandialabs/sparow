# function that returns scenario data from json file - may add additional file formats later
def scenlist_json():
    import json
    with open('scenlist.json', 'r') as file:
        data = json.load(file)
    
    return data

data = scenlist_json()  # scenario data
    
'''
bundle is a dictionary of dictionaries
    - keys are names of bundles
    - for each dictionary in bundle, keys are 'IDs' (i.e., which scenarios are in the bundle) and 'Probability'

specify which bundling scheme (function) is used via "scheme" and bundling_scheme function
'''

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
    # lower/upper bounds on the number of bundles
    min_num_buns = 4    # using min/max values from escudero paper
    max_num_buns = 8

    list_scens = data['scenarios']

    list_buns = {}
    bun_mean = [ [] for i in range(max_num_buns - min_num_buns + 1)]
    for i in range(max_num_buns - min_num_buns + 1):
        list_buns[i] = "temporary placeholder - dictionary of bundles" ### need to think about how to generalize this
        for j in range(len(list_buns)):
            bun_mean_inner = ["temporary placeholder - calculate bundle means"]
            bun_mean[i].append(bun_mean_inner) 
            
        # assign each scenario to closest bundle
        temp_dict = {}
        if all(q == bun_mean_inner[0] for q in bun_mean_inner): # skip bundle reassignment if all bundle means are the same
            temp_dict = "temporary placeholder - dictionary matching each scen to a bund"
        else:
            for k, scen in enumerate(list_scens):
                dist_scen_to_bun = [abs(scen[k]['Demand'] - bun_mean[i][j]) for j in range(len(list_buns))] 
                min_dist = min(dist_scen_to_bun) # double check index is correct here
                temp_dict[dist_scen_to_bun.index(min_dist)].append(scen)
        
        # calculate bundling error -  
        '''
        the rest of this function is pseudocode
        for j in range(len(list_buns)):
            err[i,new_bun] = sum(abs(k - bun_mean[i][new_bun]) for k in enumerate(list_scens))
    
        num_buns = argmax(sum(total_error[i,j] for j) - sum(total_error[i-1,j] for j)) ## need if condition for single bundle case
        '''


    core_scenarios = {} # core scenarios are LF (can change depending on the problem)
    for i in range(len(data['scenarios'])):
        if data['scenarios'][i]['Fidelity'] == 'LF':
            core_scenarios.append(data['scenarios'][i])


    return bundle 


def bundle_similar_cover(data): # bundle similar scenarios together; each scenario appears in two bundles
    pass 


def bundle_random_partition(data): # random bundling
    pass
###################################################################################################################


scheme = bundle_multifid
def bundle_scheme(data, scheme):
        return scheme(data)
