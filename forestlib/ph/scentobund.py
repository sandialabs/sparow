# function that returns scenario data from json file - may add additional file formats later
def scenlist_json():
    import json
    with open('scenlist.json', 'r') as file:
        data = json.load(file)
    
    return data

data = scenlist_json()  # scenario data
scheme = 0   # determines which bundling scheme to use (will probably change how we do this later)

# if-else in this file so we don't have to change class StochasticProgram
def bundle_scheme(data, scheme): # bare bones; will change as we add schema
    if scheme == 0:
        return bundle_by_fidelity(data)
    elif scheme == 1:
        return bundle_multifid(data)
    else:
        return some_other_bundling_scheme(data)
    
'''
bundle is a dictionary of dictionaries
    - keys are names of bundles
    - for each dictionary in bundle, keys are 'IDs' (i.e., which scenarios are in the bundle) and 'Probability'
'''
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
    # will sum to 1 b/c uniform distribution - will fix later with the rule from the escudero paper
    bundle['HF'] = {'IDs':         scenID_HFlist,
                    'Probability': sum(data['scenarios'][scenID_HFlist.index(j)]['Probability'] for j in scenID_HFlist)}
    bundle['LF'] = {'IDs':         scenID_LFlist,
                    'Probability': sum(data['scenarios'][scenID_LFlist.index(j)]['Probability'] for j in scenID_LFlist)}

    return bundle


def bundle_multifid(data):  # still needs some work (don't use yet!)
    bundle_names = ['Low', 'Medium', 'High']

    for bund in bundle_names:
        bundle[bund] = {}

    bundle['Low']    = {'IDs': [0, 1, 6],    'Probability': 0.2}
    bundle['Medium'] = {'IDs': [2, 3, 7, 8], 'Probability': 0.5}
    bundle['High']   = {'IDs': [4, 5, 9],    'Probability': 0.3}

    return bundle


def some_other_bundling_scheme(data):
    pass

