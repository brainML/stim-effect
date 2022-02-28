import numpy as np
import argparse
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone_generalization_format", required=True)
    parser.add_argument("--zone_residuals_format", required=True)
    parser.add_argument("--generalization_threshold", required=True)
    parser.add_argument("--residuals_threshold", required=True)
    parser.add_argument("--output", required=True)
    
    args = parser.parse_args()
    thresh_gen = args.generalization_threshold
    thresh_res = args.residuals_threshold
    print(args)
    
    # find all subject file names
    sub_generalization_fname_list = glob.glob(args.zone_generalization_format.format('*'))
    sub_residuals_fname_list = glob.glob(args.zone_residuals_format.format('*'))

    # compute average zone generalization
    zone_generalization = [np.load(sub_name, allow_pickle=True).item()['generalization'] for sub_name in sub_generalization_fname_list]
    avrg_generalization = np.mean(zone_generalization)

    # compute average zone residuals
    zone_residuals = [np.load(sub_name, allow_pickle=True).item()['residuals'] for sub_name in sub_residuals_fname_list]
    avrg_residuals = np.mean(zone_residuals)

    inferences = {}
    for i in ['A','B','C','D']:
        inferences[i] = []
            
    num_zones = zone_residuals.shape[0]
    for i in range(num_zones):
        for j in range(num_zones):
            if i == j:
                continue
            else:
                generalization = avrg_generalization[i,j]
                residual = avrg_residuals[i,j]
                
                if generalization <= thresh_gen and residual > thresh_res:
                    inferences['A'].append((i,j))
                elif generalization <= thresh_gen and residual <= thresh_res:
                    inferences['B'].append((i,j))
                elif generalization > thresh_gen and residual <= thresh_res:
                    inferences['C'].append((i,j))
                else:
                    inferences['D'].append((i,j))
    
    for key in inferences.keys():
        inferences[key] = np.vstack(inferences[key])
    
    np.save(args.output, {'inferences':inferences, 'generalization threshold':thresh_gen, 'residuals threshold':thresh_res})
