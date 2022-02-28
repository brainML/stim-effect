import numpy as np
import argparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import glob

def corr(X,Y,axis=0):
    # computes the correlation of x1 with y1, x2 with y2, and so on
    return np.mean(zscore(X,axis=axis)*zscore(Y,axis=axis),axis=axis)

# computes the pair-wise correlation of all variables in X with all variables in Y
def crosscorr(X,Y,axis=0):
    nvars_x = X.shape[-1]
    nvars_y = Y.shape[-1]

    num_samples = X.shape[0]

    rep = np.float32(np.repeat(X,nvars_y,axis=1))
    rep = np.reshape(rep, [-1, nvars_x, nvars_y])
    rep2 = np.float32(np.repeat(Y,nvars_x,axis=1))
    rep2 = np.reshape(rep2, [-1, nvars_y, nvars_x])
    rep2 = np.swapaxes(rep2, 1, 2)
    
    return corr(rep, rep2)

def get_subj_test_data(fname):
    return np.load(fname, allow_pickle=True)

def compute_residuals(subj_id1, zone_indices):
    # computes the pairwise residuals of predicting each brain zones from every other brain zone
    # returns a tensor of dimensions num brain zones x num brain zones x num samples
    test_data = get_subj_test_data(subj_id1)[:,zone_indices]
    brain_corrs = 1-squareform(pdist(test_data.T, 'correlation'))
    num_samples, num_vars = test_data.shape
    brain_residuals = np.zeros([num_vars, num_vars, num_samples])
    
    for i in range(num_vars):
        for j in range(num_vars):
            brain_residuals[i,j,:] = test_data[:,j] - brain_corrs[i,j]*test_data[:,i]
    return brain_residuals

def compute_zone_residuals(subj_id1, all_subj_ids, zone_indices):
    zone_residuals = []
    residuals_subj1 = compute_residuals(subj_id1, zone_indices)
    for subj_id2 in all_subj_ids:
        if subj_ids2 == subj_id1:
            continue
            
        residuals_subj2 = compute_residuals(subj_id2, zone_indices)
        zone_residuals.append(corr(residuals_subj1, residuals_subj2))
    zone_residuals = np.sqrt(np.mean(np.vstack(zone_residuals),0))
    return zone_residuals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_format", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--zone_indices", required=True)
    
    args = parser.parse_args()
    print(args)
    
    zone_indices = np.load(args.zone_indices)
    
    # find all subject file names
    sub_fname_list = glob.glob(args.data_file_format.format('*'))
    
    # compute zone residuals
    zone_residuals = {}
    for sub_fname in sub_fname_list:
        zone_residuals[sub_fname] = compute_zone_residuals(sub_fname, sub_fname_list, zone_indices))
    
    np.save(args.output, {'residuals':zone_residuals, zone indices':zone_indices})
