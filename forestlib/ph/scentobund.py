import json
with open('mpisppy/agnostic/scenlist.json', 'r') as file:
    data = json.load(file)

bundle = {}

def bundle_by_fidelity(data):
    bundle['HF'] = {}
    bundle['LF'] = {}
    scenID_HFlist = []
    scenID_LFlist = []
    for i in range(len(data['scenarios'])):
        if data['scenarios'][i]['Fidelity'] == 'HF':
            scenID_HFlist.append(data['scenarios'][i]['ID'])
        elif data['scenarios'][i]['Fidelity'] == 'LF':
            scenID_LFlist.append(data['scenarios'][i]['ID'])
        else:
            raise RuntimeError (f"No fidelity specified for {i}th scenario") # deal with order later

    # will sum to 1 b/c uniform distribution - fix later with the rule from the escudero paper   
    bundle['HF'] = {'IDs':         scenID_HFlist,
                    'Probability': sum(data['scenarios'][scenID_HFlist.index(j)]['Probability'] for j in scenID_HFlist),
                    'Demand':      sum(data['scenarios'][scenID_HFlist.index(j)]['Demand'] for j in scenID_HFlist)/len(scenID_HFlist)}
    bundle['LF'] = {'IDs':         scenID_LFlist,
                    'Probability': sum(data['scenarios'][scenID_LFlist.index(j)]['Probability'] for j in scenID_LFlist),
                    'Demand':      sum(data['scenarios'][scenID_LFlist.index(j)]['Demand'] for j in scenID_LFlist)/len(scenID_LFlist)}

    return bundle


def bundle_multifid(data):
    pass


bundle_by_fidelity(data)

print(bundle['LF'])

