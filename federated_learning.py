import collections
import torch

def weighted_averaging(w_list, num_sample_list):
    
    num_total_samples = sum(num_sample_list)
    keys = w_list[0].keys()
    w_avg = collections.OrderedDict()

    device = w_list[0][list(keys)[0]].device
    
    for k in keys:
        w_avg[k] = torch.zeros(w_list[0][k].size()).to(device)   # Reshape w_avg to match local weights.

    for k in keys:
        for i in range(len(w_list)):
            w_avg[k] += num_sample_list[i] * w_list[i][k]
        w_avg[k] = torch.div(w_avg[k], num_total_samples)
    return w_avg
