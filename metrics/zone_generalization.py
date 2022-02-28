import numpy as np
import argparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--zone_indices", default='')
    
    args = parser.parse_args()
    print(args)
    
    predictions = np.load(args.predictions, allow_pickle=True)
    test_data = np.load(args.data, allow_pickle=True)
    
    if args.zone_indices:
        zone_indices = np.load(args.zone_indices, allow_pickle=True)
        predictions = predictions[:,zone_indices]
        test_data = test_data[:,zone_indices]
    else:
        zone_indices = np.arange(predictions.shape[1])

    # compute the encoding model performance
    zone_generalizations = crosscorr(predictions, test_data)
    
    np.save(args.output, {'generalizations':zone_generalizations, 'zone indices':zone_indices})
