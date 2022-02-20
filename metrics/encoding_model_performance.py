import numpy as np
import argparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore

def corr(X,Y,axis=0):
    # computes the correlation of x1 with y1, x2 with y2, and so on
    return np.mean(zscore(X,axis=axis)*zscore(Y,axis=axis),axis=axis)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    
    args = parser.parse_args()
    print(args)

    predictions = np.load(args.predictions)
    test_data = np.load(args.data)

    # compute the encoding model performance
    encoding_performance = corr(predictions, test)
    
    np.save(args.output, encoding_performance)
