from scipy.linalg import toeplitz
import numpy as np
from sklearn.model_selection import KFold
from numpy.linalg import inv
from scipy.stats import zscore
import matplotlib.pyplot as plt
import time
import copy
np.random.seed(0)

def toeplitz_cov(n):
    cov = toeplitz(np.exp(-(np.arange(n*1.0))**2/n))
    return cov

def R2(Pred,Real):
    SSres = np.mean((Real-Pred)**2,0)
    SStot = np.var(Real,0)
    return np.nan_to_num(1-SSres/SStot)

def ridge(X,Y,lmbda):
    return np.dot(inv(X.T.dot(X)+lmbda*np.eye(X.shape[1])),X.T.dot(Y))

def ridge_by_lambda(X, Y, Xval, Yval, lambdas):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def cross_val_ridge(train_features, train_data, n_splits=10, lambdas=np.array([10**i for i in range(-6, 10)])):
    n_voxels = train_data.shape[1]
    n_lambdas = lambdas.shape[0]
    n_feats = train_features.shape[1]
    r_cv = np.zeros((n_lambdas, n_voxels))
    
    kf = KFold(n_splits=n_splits)
    for icv, (trn, val) in enumerate(kf.split(train_data)):
        cost = ridge_by_lambda(train_features[trn], train_data[trn], train_features[val], train_data[val], lambdas=lambdas)
        r_cv += cost
        
    argmin_lambda = np.argmin(r_cv,axis = 0)
    weights = np.zeros((n_feats,n_voxels))
    for idx_lambda in range(lambdas.shape[0]):
        idx_vox = argmin_lambda == idx_lambda
        weights[:,idx_vox] = ridge(train_features, train_data[:,idx_vox],lambdas[idx_lambda])
    min_lambdas = np.array([lambdas[i] for i in argmin_lambda])

    return weights, min_lambdas

def corr(X,Y):
    return np.mean(zscore(X)*zscore(Y),0)

def get_plot_ys(corrs, y_keys):
    # Input: expects corrs to have repetition data stored in 
    # corrs['v1v2'][k] and/or corrs['v1p1'][k] and/or corrs['v2p1'][k] for k in y_keys.
    # Output: ys and ys_std with computed means and stddevs for each k for easy error bar plotting.
    ys, ys_std = {}, {}
    for corr_type in corrs.keys(): # 'v1v2'/'v1p1'/'v2p1'
        ys[corr_type], ys_std[corr_type] = [], []
    
    # Compute means and stddevs for easy error bar plotting
    for k in y_keys:
        for corr_type in corrs.keys(): # 'v1v2'/'v1p1'/'v2p1'
            ys[corr_type].append(np.mean(np.array(corrs[corr_type][k])))
            ys_std[corr_type].append(np.std(np.array(corrs[corr_type][k])))
    return ys, ys_std

def generate_features_nonfeat_stimulus(n, ms):
    m1, m2, m3, m12 = ms
    # Features
    x1 = np.random.multivariate_normal(mean=np.zeros(m1), cov=toeplitz_cov(m1), size=(n,)) if m1>0 else np.array([]).reshape(n,0)
    x2 = np.random.multivariate_normal(mean=np.zeros(m2), cov=toeplitz_cov(m2), size=(n,)) if m2>0 else np.array([]).reshape(n,0)
    x3 = np.random.multivariate_normal(mean=np.zeros(m3), cov=toeplitz_cov(m3), size=(n,)) if m3>0 else np.array([]).reshape(n,0)
    x12 = np.random.multivariate_normal(mean=np.zeros(m12), cov=toeplitz_cov(m12), size=(n,)) if m12>0 else np.array([]).reshape(n,0)
    x = np.concatenate((x1, x2, x3, x12), axis=1) # x: (n, m1+m2+m3+m12)
    
    # Stimulus - Features
    z1 = np.random.multivariate_normal(mean=np.zeros(m1), cov=toeplitz_cov(m1), size=(n,)) if m1>0 else np.array([]).reshape(n,0)
    z2 = np.random.multivariate_normal(mean=np.zeros(m2), cov=toeplitz_cov(m2), size=(n,)) if m2>0 else np.array([]).reshape(n,0)
    z3 = np.random.multivariate_normal(mean=np.zeros(m3), cov=toeplitz_cov(m3), size=(n,)) if m3>0 else np.array([]).reshape(n,0)
    z12 = np.random.multivariate_normal(mean=np.zeros(m12), cov=toeplitz_cov(m12), size=(n,)) if m12>0 else np.array([]).reshape(n,0)
    z = np.concatenate((z1, z2, z3, z12), axis=1) # z: (n, m1+m2+m3+m12)
    
    return (x, x1, x2, x3, x12), (z, z1, z2, z3, z12)

def generate_weights_nonfeat_stimulus(n, ms):
    m1, m2, m3, m12 = ms
    
    # Feature Weights
    w1 = np.random.rand(m1, 1)
    w2 = np.random.rand(m2, 1)
    w12 = np.random.rand(m12, 1)
    
    per_subj_weights_noise = 0.25
    subj1_w1 = np.random.normal(loc=w1, scale=np.ones((m1, 1))*per_subj_weights_noise)
    subj1_w2 = np.random.normal(loc=w2, scale=np.ones((m2, 1))*per_subj_weights_noise)
    subj1_w12 = np.random.normal(loc=w12, scale=np.ones((m12, 1))*per_subj_weights_noise)
    
    subj2_w1 = np.random.normal(loc=w1, scale=np.ones((m1, 1))*per_subj_weights_noise)
    subj2_w2 = np.random.normal(loc=w2, scale=np.ones((m2, 1))*per_subj_weights_noise)
    subj2_w12 = np.random.normal(loc=w12, scale=np.ones((m12, 1))*per_subj_weights_noise)
    
    # Stimulus - Feature Weights
    u1 = np.random.rand(m1, 1)
    u2 = np.random.rand(m2, 1)
    u12 = np.random.rand(m12, 1)
    
    subj1_u1 = np.random.normal(loc=u1, scale=np.ones((m1, 1))*per_subj_weights_noise)
    subj1_u2 = np.random.normal(loc=u2, scale=np.ones((m2, 1))*per_subj_weights_noise)
    subj1_u12 = np.random.normal(loc=u12, scale=np.ones((m12, 1))*per_subj_weights_noise)
    
    subj2_u1 = np.random.normal(loc=u1, scale=np.ones((m1, 1))*per_subj_weights_noise)
    subj2_u2 = np.random.normal(loc=u2, scale=np.ones((m2, 1))*per_subj_weights_noise)
    subj2_u12 = np.random.normal(loc=u12, scale=np.ones((m12, 1))*per_subj_weights_noise)
    
    return ((subj1_w1, subj1_w2, subj1_w12), (subj1_u1, subj1_u2, subj1_u12)), ((subj2_w1, subj2_w2, subj2_w12), (subj2_u1, subj2_u2, subj2_u12))

def generate_voxel_from_params_with_all_hyperparams(xs, ws, zs, us, epss, relevant_hyperparams):
    (xi, x12), (wi, w12), (zi, z12), (ui, u12), (epsi, eps12) = xs, ws, zs, us, epss
    alpha, beta_i, gamma = relevant_hyperparams
    vi_feat_related_component = alpha*zscore(np.dot(x12, w12)) + (1-alpha)*zscore(np.dot(xi, wi))
    vi_feat_unrelated_component = alpha*(zscore(gamma*zscore(np.dot(zi, ui)) + (1-gamma)*epsi)) + (1-alpha)*(zscore(gamma*zscore(np.dot(z12, u12)) + (1-gamma)*eps12))
    vi = beta_i*vi_feat_related_component + (1-beta_i)*vi_feat_unrelated_component
    return vi

def generate_voxels_with_all_hyperparams_nonfeat_stimulus(n, all_stimulus_information, all_ground_truth_weights, hyperparams):
    # Initialize noise
    eps_std1, eps_std2, eps_std12 = 1.0, 1.0, 1.0 # As mentioned in NOTE above.
    subj1_eps1 = np.random.normal(loc=0.0, scale=eps_std1, size=(n, 1))
    subj1_eps2 = np.random.normal(loc=0.0, scale=eps_std2, size=(n, 1))
    subj1_eps12 = np.random.normal(loc=0.0, scale=eps_std12, size=(n, 1))
    
    subj2_eps1 = np.random.normal(loc=0.0, scale=eps_std1, size=(n, 1))
    subj2_eps2 = np.random.normal(loc=0.0, scale=eps_std2, size=(n, 1))
    subj2_eps12 = np.random.normal(loc=0.0, scale=eps_std12, size=(n, 1))
    
    # Generate voxel data
    (x, x1, x2, x3, x12), (z, z1, z2, z3, z12) = all_stimulus_information
    ((subj1_w1, subj1_w2, subj1_w12), (subj1_u1, subj1_u2, subj1_u12)), ((subj2_w1, subj2_w2, subj2_w12), (subj2_u1, subj2_u2, subj2_u12)) = all_ground_truth_weights
    alpha, beta_1, beta_2, gamma = hyperparams
    
    subj1_v1 = generate_voxel_from_params_with_all_hyperparams((x1, x12), (subj1_w1, subj1_w12), (z1, z12), (subj1_u1, subj1_u12), (subj1_eps1, subj1_eps12), (alpha, beta_1, gamma))
    subj1_v2 = generate_voxel_from_params_with_all_hyperparams((x2, x12), (subj1_w2, subj1_w12), (z2, z12), (subj1_u2, subj1_u12), (subj1_eps2, subj1_eps12), (alpha, beta_2, gamma))
    
    subj2_v1 = generate_voxel_from_params_with_all_hyperparams((x1, x12), (subj2_w1, subj1_w12), (z1, z12), (subj2_u1, subj2_u12), (subj2_eps1, subj2_eps12), (alpha, beta_1, gamma))
    subj2_v2 = generate_voxel_from_params_with_all_hyperparams((x2, x12), (subj2_w2, subj1_w12), (z2, z12), (subj2_u2, subj2_u12), (subj2_eps2, subj2_eps12), (alpha, beta_2, gamma))                           
    return (subj1_v1, subj1_v2), (subj2_v1, subj2_v2)

def perform_experiment_with_all_hyperparams_nonfeat_stimulus(n, hyperparams):
    # Generate features, weights, voxel data
    ms = [10, 10, 10, 10] # NOTE: Same ms are used for xs and zs (S - X components)
    all_stimulus_information = generate_features_nonfeat_stimulus(n, ms) # contains xs and zs
    all_ground_truth_weights = generate_weights_nonfeat_stimulus(n, ms)
    (subj1_v1, subj1_v2), (subj2_v1, subj2_v2) = generate_voxels_with_all_hyperparams_nonfeat_stimulus(n, all_stimulus_information, all_ground_truth_weights, hyperparams)

    # Estimate weights using ridge
    x = all_stimulus_information[0][0]
    subj1_v1_est_weight, _ = cross_val_ridge(x, subj1_v1)
    subj1_v2_est_weight, _ = cross_val_ridge(x, subj1_v2)
    subj2_v1_est_weight, _ = cross_val_ridge(x, subj2_v1)
    subj2_v2_est_weight, _ = cross_val_ridge(x, subj2_v2)
    
    # Compute predictions
    subj1_p1 = np.dot(x, subj1_v1_est_weight)
    subj1_p2 = np.dot(x, subj1_v2_est_weight)
    subj2_p1 = np.dot(x, subj2_v1_est_weight)
    subj2_p2 = np.dot(x, subj2_v2_est_weight)
    
    # Compute residuals for v1
    subj1_v2tov1_estimated_weight, _ = cross_val_ridge(subj1_v2, subj1_v1)
    subj1_v1_residual = subj1_v1 - np.dot(subj1_v2, subj1_v2tov1_estimated_weight)
    
    subj2_v2tov1_estimated_weight, _ = cross_val_ridge(subj2_v2, subj2_v1)
    subj2_v1_residual = subj2_v1 - np.dot(subj2_v2, subj2_v2tov1_estimated_weight)
                
    # Compute relevant correlations 
    # func_conn,encoding_model_perf,normalized_gen_perf use only subj1 data
    # corr_residuals uses only v1 data
    func_conn = corr(subj1_v1, subj1_v2)
    encoding_model_perf = corr(subj1_v1, subj1_p1)
    normalized_gen_perf = corr(subj1_v2, subj1_p1)/corr(subj1_v2, subj1_p2)
    gen_perf = corr(subj1_v2, subj1_p1)
    corr_residuals = corr(subj1_v1_residual, subj2_v1_residual)
    return func_conn, encoding_model_perf, normalized_gen_perf, corr_residuals, gen_perf

###############################
### PERFORMING SIMULATIONS ####
###############################
###############################
# VARYING ALPHA. FIXED DELTA. #
###############################
all_alphas = [0.1*i for i in range(11)]
n = 400 # no. of train samples
n_repetitions = 1000

corrs = {}
corrs['func_conn'], corrs['encoding_perf'], corrs['norm_gen_perf'], corrs['corr_residuals'], corrs['gen_perf'] = {}, {}, {}, {}, {}

beta_1, beta_2, gamma = 0.5, 0.5, 1.0
for alpha in all_alphas:
    print("alpha={}".format(alpha))
    corrs['func_conn'][alpha], corrs['encoding_perf'][alpha], corrs['norm_gen_perf'][alpha], corrs['corr_residuals'][alpha], corrs['gen_perf'][alpha] = [], [], [], [], []
    for i in range(n_repetitions):
        func_conn, encoding_model_perf, normalized_gen_perf, corr_residuals, gen_perf = perform_experiment_with_all_hyperparams_nonfeat_stimulus(n, (alpha, beta_1, beta_2, gamma))
        corrs['func_conn'][alpha].append(func_conn)
        corrs['encoding_perf'][alpha].append(encoding_model_perf)
        corrs['norm_gen_perf'][alpha].append(normalized_gen_perf)
        corrs['corr_residuals'][alpha].append(corr_residuals)
        corrs['gen_perf'][alpha].append(gen_perf)

corrs_alpha = copy.deepcopy(corrs)
###############################
# VARYING DELTA. FIXED ALPHA. #
###############################
all_gammas = [0.1*i for i in range(11)]
n = 400 # no. of train samples
n_repetitions = 1000

corrs = {}
corrs['func_conn'], corrs['encoding_perf'], corrs['norm_gen_perf'], corrs['corr_residuals'], corrs['gen_perf'] = {}, {}, {}, {}, {}

beta_1, beta_2, alpha = 0.5, 0.5, 1.0
for gamma in all_gammas:
    print("gamma={}".format(gamma))
    corrs['func_conn'][gamma], corrs['encoding_perf'][gamma], corrs['norm_gen_perf'][gamma], corrs['corr_residuals'][gamma], corrs['gen_perf'][gamma] = [], [], [], [], []
    for i in range(n_repetitions):
        func_conn, encoding_model_perf, normalized_gen_perf, corr_residuals, gen_perf = perform_experiment_with_all_hyperparams_nonfeat_stimulus(n, (alpha, beta_1, beta_2, gamma))
        corrs['func_conn'][gamma].append(func_conn)
        corrs['encoding_perf'][gamma].append(encoding_model_perf)
        corrs['norm_gen_perf'][gamma].append(normalized_gen_perf)
        corrs['corr_residuals'][gamma].append(corr_residuals)
        corrs['gen_perf'][gamma].append(gen_perf)

corrs_gamma = copy.deepcopy(corrs)

########################
# VARYING ALPHA, DELTA. #
########################
all_alphas = [0.1*i for i in range(11)]
all_gammas = [0.1*i for i in range(11)]
n = 400 # no. of train samples
n_repetitions = 100

corrs = {}
corrs['func_conn'], corrs['encoding_perf'], corrs['norm_gen_perf'], corrs['corr_residuals'], corrs['gen_perf'] = {}, {}, {}, {}, {}

beta_1, beta_2 = 0.5, 0.5
for alpha in all_alphas:
    print("alpha={}".format(alpha))
    corrs['func_conn'][alpha], corrs['encoding_perf'][alpha], corrs['norm_gen_perf'][alpha], corrs['corr_residuals'][alpha], corrs['gen_perf'][alpha] = {}, {}, {}, {}, {}
    for gamma in all_gammas:
        corrs['func_conn'][alpha][gamma], corrs['encoding_perf'][alpha][gamma], corrs['norm_gen_perf'][alpha][gamma], corrs['corr_residuals'][alpha][gamma], corrs['gen_perf'][alpha][gamma] = [], [], [], [], []
        for i in range(n_repetitions):
            func_conn, encoding_model_perf, normalized_gen_perf, corr_residuals, gen_perf = perform_experiment_with_all_hyperparams_nonfeat_stimulus(n, (alpha, beta_1, beta_2, gamma))
            corrs['func_conn'][alpha][gamma].append(func_conn)
            corrs['encoding_perf'][alpha][gamma].append(encoding_model_perf)
            corrs['norm_gen_perf'][alpha][gamma].append(normalized_gen_perf)
            corrs['corr_residuals'][alpha][gamma].append(corr_residuals)
            corrs['gen_perf'][alpha][gamma].append(gen_perf)

corrs_vary_alpha_all_gammas = copy.deepcopy(corrs)
np.save('spatial_generalization_experiments/appendix_results/varying_alpha_all_gamma_pt2.npy', corrs_vary_alpha_all_gammas)

######################################
# VARYING ALPHA, BETAS. FIXED DELTA. #
######################################
all_alphas = [0.25*i for i in range(5)]
all_betas = [0.2*i for i in range(6)]
n = 400 # no. of train samples
n_repetitions = 100

corrs = {}
corrs['func_conn'], corrs['encoding_perf'], corrs['norm_gen_perf'], corrs['corr_residuals'], corrs['gen_perf'] = {}, {}, {}, {}, {}

gamma = 1.0
for alpha in all_alphas:
    print("alpha={}".format(alpha))
    corrs['func_conn'][alpha], corrs['encoding_perf'][alpha], corrs['norm_gen_perf'][alpha], corrs['corr_residuals'][alpha], corrs['gen_perf'][alpha] = {}, {}, {}, {}, {}
    for beta_1 in all_betas:
        corrs['func_conn'][alpha][beta_1], corrs['encoding_perf'][alpha][beta_1], corrs['norm_gen_perf'][alpha][beta_1], corrs['corr_residuals'][alpha][beta_1], corrs['gen_perf'][alpha][beta_1] = {}, {}, {}, {}, {}
        for beta_2 in all_betas:
            corrs['func_conn'][alpha][beta_1][beta_2], corrs['encoding_perf'][alpha][beta_1][beta_2], corrs['norm_gen_perf'][alpha][beta_1][beta_2], corrs['corr_residuals'][alpha][beta_1][beta_2], corrs['gen_perf'][alpha][beta_1][beta_2] = [], [], [], [], []
            for i in range(n_repetitions):
                func_conn, encoding_model_perf, normalized_gen_perf, corr_residuals, gen_perf = perform_experiment_with_all_hyperparams_nonfeat_stimulus(n, (alpha, beta_1, beta_2, gamma))
                corrs['func_conn'][alpha][beta_1][beta_2].append(func_conn)
                corrs['encoding_perf'][alpha][beta_1][beta_2].append(encoding_model_perf)
                corrs['norm_gen_perf'][alpha][beta_1][beta_2].append(normalized_gen_perf)
                corrs['corr_residuals'][alpha][beta_1][beta_2].append(corr_residuals)
                corrs['gen_perf'][alpha][beta_1][beta_2].append(gen_perf)

corrs_vary_alpha_all_betas_gamma1 = copy.deepcopy(corrs)
np.save('spatial_generalization_experiments/appendix_results/varying_alpha_all_beta1_beta2_pt2_gamma1.npy', corrs_vary_alpha_all_betas_gamma1)

######################################
# VARYING DELTA, BETAS. FIXED ALPHA. #
######################################
all_gammas = [0.25*i for i in range(5)]
all_betas = [0.2*i for i in range(6)]
n = 400 # no. of train samples
n_repetitions = 100

corrs = {}
corrs['func_conn'], corrs['encoding_perf'], corrs['norm_gen_perf'], corrs['corr_residuals'], corrs['gen_perf'] = {}, {}, {}, {}, {}

alpha = 1.0
for gamma in all_gammas:
    print("gamma={}".format(gamma))
    corrs['func_conn'][gamma], corrs['encoding_perf'][gamma], corrs['norm_gen_perf'][gamma], corrs['corr_residuals'][gamma], corrs['gen_perf'][gamma] = {}, {}, {}, {}, {}
    for beta_1 in all_betas:
        corrs['func_conn'][gamma][beta_1], corrs['encoding_perf'][gamma][beta_1], corrs['norm_gen_perf'][gamma][beta_1], corrs['corr_residuals'][gamma][beta_1], corrs['gen_perf'][gamma][beta_1] = {}, {}, {}, {}, {}
        for beta_2 in all_betas:
            corrs['func_conn'][gamma][beta_1][beta_2], corrs['encoding_perf'][gamma][beta_1][beta_2], corrs['norm_gen_perf'][gamma][beta_1][beta_2], corrs['corr_residuals'][gamma][beta_1][beta_2], corrs['gen_perf'][gamma][beta_1][beta_2] = [], [], [], [], []
            for i in range(n_repetitions):
                func_conn, encoding_model_perf, normalized_gen_perf, corr_residuals, gen_perf = perform_experiment_with_all_hyperparams_nonfeat_stimulus(n, (alpha, beta_1, beta_2, gamma))
                corrs['func_conn'][gamma][beta_1][beta_2].append(func_conn)
                corrs['encoding_perf'][gamma][beta_1][beta_2].append(encoding_model_perf)
                corrs['norm_gen_perf'][gamma][beta_1][beta_2].append(normalized_gen_perf)
                corrs['corr_residuals'][gamma][beta_1][beta_2].append(corr_residuals)
                corrs['gen_perf'][gamma][beta_1][beta_2].append(gen_perf)

corrs_vary_gamma_all_betas = copy.deepcopy(corrs)
np.save('spatial_generalization_experiments/appendix_results/varying_gamma_all_beta1_beta2_pt2_anybeta.npy', corrs_vary_gamma_all_betas)

####################################
### PLOTTING SIMULATION RESULTS ####
####################################
###################
# MAIN TEXT PLOTS #
###################
plt.rcParams.update({'font.size': 30})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20))
ys_alpha, ys_std_alpha = get_plot_ys(corrs_alpha, all_alphas)
l1 = ax1.errorbar(all_alphas, ys_alpha['encoding_perf'], yerr=ys_std_alpha['encoding_perf'], fmt='o', label='Encoding model performance', markersize=13., color='#1e90ff')
l2 = ax1.errorbar(all_alphas, ys_alpha['func_conn'], yerr=ys_std_alpha['func_conn'], fmt='^', label='Source correlation', markersize=14., color='#ff9f00')
l3 = ax1.errorbar(all_alphas, ys_alpha['gen_perf'], yerr=ys_std_alpha['gen_perf'], fmt='s', label='Source generalization', markersize=12., color='#ee82ee')
l4 = ax1.errorbar(all_alphas, ys_alpha['corr_residuals'], yerr=ys_std_alpha['corr_residuals'], fmt='D', label='Source residuals', markersize=12., color='#6ea058')

ys_gamma, ys_std_gamma = get_plot_ys(corrs_gamma, all_gammas)
ax2.errorbar(all_gammas, ys_gamma['encoding_perf'], yerr=ys_std_gamma['encoding_perf'], fmt='o', label='Encoding model performance', markersize=13., color='#1e90ff')
ax2.errorbar(all_gammas, ys_gamma['func_conn'], yerr=ys_std_gamma['func_conn'], fmt='^', label='Source correlation', markersize=14., color='#ff9f00')
ax2.errorbar(all_gammas, ys_gamma['gen_perf'], yerr=ys_std_gamma['gen_perf'], fmt='s', label='Source generalization', markersize=12., color='#ee82ee')
ax2.errorbar(all_gammas, ys_gamma['corr_residuals'], yerr=ys_std_gamma['corr_residuals'], fmt='D', label='Source residuals', markersize=12., color='#6ea058')

ax1.set_xlabel(r'$\alpha$', fontsize=40, labelpad=15)
ax1.set_ylabel('metric', fontsize=35, labelpad=15)
ax1.set_aspect('equal')
ax1.grid(linestyle='--', linewidth=1, alpha=0.3)
ax1.set_xlim([-0.05,1.02])
ax1.set_ylim([-0.05,1.02])

ax2.set_xlabel(r'$\delta$', fontsize=40, labelpad=15)
ax2.set_ylabel('metric', fontsize=35, labelpad=15)
ax2.set_aspect('equal')
ax2.grid(linestyle='--', linewidth=1, alpha=0.3)
ax2.set_xlim([-0.05,1.02])
ax2.set_ylim([-0.05,1.02])

fig.legend(handles = [l1,l2,l3,l4] , labels=['Encoding model performance', 'Source correlation', 'Source generalization', 'Source residuals'],loc='upper center', 
             bbox_to_anchor=(0.5, 0.25),fancybox=False, shadow=False, ncol=2, fontsize=32, handletextpad=0.1)

plt.tight_layout(w_pad=4)
plt.savefig('{}.pdf'.format('spatial_generalization_experiments/figures/temp'))

##################
# APPENDIX PLOTS #
##################
plt.rcParams.update({'font.size': 15})
############################## FIG 2A - VARYING DELTA ###########################
############################## FIG 2B - VARYING ALPHA ###########################
all_alphas = [0.1*i for i in range(11)]
all_gammas = [0.1*i for i in range(11)]
X, Y, Z_func_conn, Z_encoding_perf, Z_norm_gen_perf, Z_corr_residuals = np.zeros((11,11)), np.zeros((11,11)), np.zeros((11,11)), np.zeros((11,11)), np.zeros((11,11)), np.zeros((11,11))

fig, axs = plt.subplots(1, 4, figsize=(20,5))

for i, alpha in enumerate(all_alphas):
    for j, gamma in enumerate(all_gammas):
        X[i][j] = alpha
        Y[i][j] = gamma
        Z_func_conn[i][j] = np.mean(np.array(corrs_vary_alpha_all_gammas['func_conn'][alpha][gamma]))
        Z_encoding_perf[i][j] = np.mean(np.array(corrs_vary_alpha_all_gammas['encoding_perf'][alpha][gamma]))
        Z_norm_gen_perf[i][j] = np.mean(np.array(corrs_vary_alpha_all_gammas['norm_gen_perf'][alpha][gamma]))
        Z_corr_residuals[i][j] = np.mean(np.array(corrs_vary_alpha_all_gammas['corr_residuals'][alpha][gamma]))

axs[0].contourf(X, Y, Z_func_conn, cmap='RdGy_r', vmin=0, vmax=1, levels=[0.1*i for i in range(11)], extend='both')
axs[0].set_title('Functional connectivity')
axs[0].set_xticks([0.25*i for i in range(5)])
axs[0].set_yticks([0.25*i for i in range(1,5)])
axs[0].set_aspect('equal')

axs[1].contourf(X, Y, Z_encoding_perf, cmap='RdGy_r', vmin=0, vmax=1, levels=[0.1*i for i in range(11)], extend='both')
axs[1].set_title('Encoding model performance')
axs[1].set_xticks([0.25*i for i in range(5)])
axs[1].set_yticks([0.25*i for i in range(1,5)])
axs[1].set_aspect('equal')

axs[2].contourf(X, Y, Z_norm_gen_perf, cmap='RdGy_r', vmin=0, vmax=1, levels=[0.1*i for i in range(11)], extend='both')
axs[2].set_title('Source generalization')
axs[2].set_xticks([0.25*i for i in range(5)])
axs[2].set_yticks([0.25*i for i in range(1,5)])
axs[2].set_aspect('equal')

ax3 = axs[3].contourf(X, Y, Z_corr_residuals, cmap='RdGy_r', vmin=0, vmax=1, levels=[0.1*i for i in range(11)], extend='both')
axs[3].set_title('Source residuals')
axs[3].set_xticks([0.25*i for i in range(5)])
axs[3].set_yticks([0.25*i for i in range(1,5)])
axs[3].set_aspect('equal')

fig.colorbar(ax3, ax=axs[-1])
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(r'$\alpha$', fontsize=30)
plt.ylabel(r'$\delta$', fontsize=30)

plt.tight_layout()
plt.savefig('{}.pdf'.format('spatial_generalization_experiments/figures/appendix_fig1'))

############################## FIG 2A - VARYING BETAS AT DELTA=1.0 ###########################
all_alphas = [0.25*i for i in range(5)]
all_betas = [0.2*i for i in range(6)]

fig, axs = plt.subplots(4, 5, figsize=(20,15))

for k, alpha in enumerate(all_alphas):
    X, Y, Z_func_conn, Z_encoding_perf, Z_gen_perf, Z_corr_residuals = np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6))
    for i, beta_1 in enumerate(all_betas):
        for j, beta_2 in enumerate(all_betas):
            X[i][j] = beta_1
            Y[i][j] = beta_2
            Z_func_conn[i][j] = np.mean(np.array(corrs_vary_alpha_all_betas['func_conn'][alpha][beta_1][beta_2]))
            Z_encoding_perf[i][j] = np.mean(np.array(corrs_vary_alpha_all_betas['encoding_perf'][alpha][beta_1][beta_2]))
            Z_gen_perf[i][j] = np.mean(np.array(corrs_vary_alpha_all_betas['gen_perf'][alpha][beta_1][beta_2]))
            Z_corr_residuals[i][j] = np.mean(np.array(corrs_vary_alpha_all_betas['corr_residuals'][alpha][beta_1][beta_2]))
            
    axs[0][k].contourf(X, Y, Z_func_conn, cmap='RdGy_r', vmin=0, vmax=1, levels=[0.1*i for i in range(11)], extend='both')
    axs[1][k].contourf(X, Y, Z_encoding_perf, cmap='RdGy_r', vmin=0, vmax=1, levels=[0.1*i for i in range(11)], extend='both')
    axs[2][k].contourf(X, Y, Z_gen_perf, cmap='RdGy_r', vmin=0, vmax=1, levels=[0.1*i for i in range(11)], extend='both')
    ax3 = axs[3][k].contourf(X, Y, Z_corr_residuals, cmap='RdGy_r', vmin=0, vmax=1, levels=[0.1*i for i in range(11)], extend='both')
    
    axs[0][k].set_xticks([0.5*i for i in range(3)])
    axs[0][k].set_yticks([0.5*i for i in range(1,3)])
    axs[1][k].set_xticks([0.5*i for i in range(3)])
    axs[1][k].set_yticks([0.5*i for i in range(1,3)])
    axs[2][k].set_xticks([0.5*i for i in range(3)])
    axs[2][k].set_yticks([0.5*i for i in range(1,3)])
    axs[3][k].set_xticks([0.5*i for i in range(3)])
    axs[3][k].set_yticks([0.5*i for i in range(1,3)])

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(r'$\beta_{1}$', fontsize=25, labelpad=5)
plt.ylabel(r'$\beta_{2}$', fontsize=25, labelpad=5)
plt.title(r'$\alpha$', fontsize=50, pad=35)

pad = 5
cols = [r'$\alpha={}$'.format(0.25*col) for col in range(5)]
rows = ['{}'.format(row) for row in ['Functional\nconnectivity', 'Encoding model\nperformance', 'Source\ngeneralization', 'Source\nresidual']]
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, 10),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline', fontsize=20)

for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
    
fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.0)

plt.tight_layout()
plt.savefig('{}.pdf'.format('spatial_generalization_experiments/figures/appendix_fig2'))

############################## FIG 2B - VARYING BETAS AT ALPHA=1.0 ###########################
all_gammas = [0.25*i for i in range(5)]
all_betas = [0.2*i for i in range(6)]

fig, axs = plt.subplots(4, 5, figsize=(20,15))

for k, gamma in enumerate(all_gammas):
    X, Y, Z_func_conn, Z_encoding_perf, Z_gen_perf, Z_corr_residuals = np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6))
    for i, beta_1 in enumerate(all_betas):
        for j, beta_2 in enumerate(all_betas):
            X[i][j] = beta_1
            Y[i][j] = beta_2
            Z_func_conn[i][j] = np.mean(np.array(corrs_vary_gamma_all_betas['func_conn'][gamma][beta_1][beta_2]))
            Z_encoding_perf[i][j] = np.mean(np.array(corrs_vary_gamma_all_betas['encoding_perf'][gamma][beta_1][beta_2]))
            Z_gen_perf[i][j] = np.mean(np.array(corrs_vary_gamma_all_betas['gen_perf'][gamma][beta_1][beta_2]))
            Z_corr_residuals[i][j] = np.mean(np.array(corrs_vary_gamma_all_betas['corr_residuals'][gamma][beta_1][beta_2]))
            
    axs[0][k].contourf(X, Y, Z_func_conn, cmap='RdGy_r', vmin=0, vmax=1, levels=[0.1*i for i in range(11)], extend='both')
    axs[1][k].contourf(X, Y, Z_encoding_perf, cmap='RdGy_r', vmin=0, vmax=1, levels=[0.1*i for i in range(11)], extend='both')
    axs[2][k].contourf(X, Y, Z_gen_perf, cmap='RdGy_r', vmin=0, vmax=1, levels=[0.1*i for i in range(11)], extend='both')
    axs[3][k].contourf(X, Y, Z_corr_residuals, cmap='RdGy_r', vmin=0, vmax=1, levels=[0.1*i for i in range(11)], extend='both')
    
    axs[0][k].set_xticks([0.5*i for i in range(3)])
    axs[0][k].set_yticks([0.5*i for i in range(1,3)])
    axs[1][k].set_xticks([0.5*i for i in range(3)])
    axs[1][k].set_yticks([0.5*i for i in range(1,3)])
    axs[2][k].set_xticks([0.5*i for i in range(3)])
    axs[2][k].set_yticks([0.5*i for i in range(1,3)])
    axs[3][k].set_xticks([0.5*i for i in range(3)])
    axs[3][k].set_yticks([0.5*i for i in range(1,3)])
    
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel(r'$\beta_{1}$', fontsize=25, labelpad=5)
plt.ylabel(r'$\beta_{2}$', fontsize=25, labelpad=5)
plt.title(r'$\delta$', fontsize=50, pad=35)

pad = 5
cols = [r'$\delta={}$'.format(0.25*col) for col in range(5)]
rows = ['{}'.format(row) for row in ['Functional\nconnectivity', 'Encoding model\nperformance', 'Source\ngeneralization', 'Source\nresidual']]
for ax, col in zip(axs[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, 10),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline', fontsize=20)

for ax, row in zip(axs[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
    
fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.0)

plt.tight_layout()
plt.savefig('{}.pdf'.format('spatial_generalization_experiments/figures/appendix_fig3'))
