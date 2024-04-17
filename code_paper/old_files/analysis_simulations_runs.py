# Author: Sara Sommariva <sara.sommariva@aalto.fi>

# Objective: Simulate synthetic data.

import os.path as op
import os
import numpy as np
from scipy import signal
import itertools

import pickle

from mne.connectivity import spectral_connectivity
from mne import (read_forward_solution, pick_types_forward, 
                 convert_forward_solution, read_labels_from_annot, 
                 SourceEstimate, extract_label_time_course)

import sys
sys.path.insert(0, '../code/')
import f_inverse
import flame


do_simulated_tc = False # Before running the simulation, the time-courses of interest have to be generated
subjects = ['k1_T1', 'k2_T1', 'k4_T1', 'k6_T1', 'k7_T1', 
           'CC110045', 'CC110182', 'CC110174', 'CC110126', 'CC110056']

# Trick for running all simulations with just one script.
num_subs = len(subjects)
if not do_simulated_tc:
    job_run = int(sys.argv[1]) - 1
    i_run = job_run // num_subs
    subject_id = job_run%num_subs

# In[]: Step 1. Define general parameters
subjects_dir = op.join('/m/nbe/work/sommars1/FLAME/', 'subjects_flame')
string_lf = op.join('./data', 
                  '{:s}_meg_single_layer-fwd.fif')
folder_fl = op.join('./flame_parcellations')
string_target_file = op.join(folder_fl,
                        '{:s}_flame_grad_k{:d}_gamma{:1.2f}_theta{:1.2f}.pkl')

folder_results = op.join('./test_simulation')
target_file_xint = op.join(folder_results, 'x_mvar.pkl')
folder_results_sub = op.join(folder_results, '{:s}')
string_results = op.join(folder_results_sub, 
                         'result_grad_num_{:d}_{:s}.pkl')

for subject in subjects:
    if not op.isdir(folder_results_sub.format(subject)):
        print('Creating folder %s'%folder_results_sub.format(subject))
        os.mkdir(op.join(folder_results_sub.format(subject)))
        
sensors_meg = 'grad' # True or'grad'

# 1.1. Parameters for flame analysis
gamma_tot = np.arange(0, 1.01, 0.2)
knn_tot = [10, 20, 30, 40]
theta = 0.05
parc = 'aparc' 

# 1.2. Parameter for generating the time-courses of the interacting sources
ndip = 2
T  = 60
ft = 250
time = np.arange(0, T, 1/ft)
f_min = 8
f_max = 13

# 1.3. Parameters for forward modeling 
alpha_bio_tot = [0.25, 0.5]
snr = 3

# 1.4. Parameters for inverse modeling
lam = 1/snr
depth = None
method = 'dSPM'
method_cp = 'mean_flip'

# 1.5. Parameters for statistical test
N_surr = 100
alpha = 0.95
idx_alpha = int(np.floor(N_surr * alpha)) - 1
method_surr = 'random_phase'

# In[]: Step 2. Define time-courses of the interacting sources
if do_simulated_tc: 

    # 2.1. Define and store seed
    seed = np.random.randint(0, 2**32-1)
    np.random.seed(seed)
    
    # 2.2. Define mvar parameters
    K = 1
    A = np.array([np.array([0.5, 0]), np.array([0.7, 0.2])])
    sigma_p = 1
    
    lambda_max = np.max(abs(np.linalg.eigvals(A)))
    if lambda_max >= 1:
        print('Warning: Unstable MVAR')
    
    # 2.2. Generate mvar process
    N0 = 1000
    Nt = time.shape[0]+N0
    eps = np.random.randn(ndip, Nt)
    x_tmp = eps
    for i in np.arange(K, time.shape[0]+N0, 1):
        x_tmp[:, i] = A.dot(x_tmp[:, i-1]) + eps[:, i]
    x_nofilter = x_tmp[:, N0:]
    
    # 2.3. Band-pass filter the signal
    band = 2/ft * np.array([f_min, f_max])
    [b, a] = signal.butter(3, band, btype='bandpass')
    x = signal.filtfilt(b, a, x_nofilter)
    
    # 2.4. Compute connectivity in the alpha-band
    x_demean = x - np.mean(x, axis=1)[:, None]
    x_ep = f_inverse.window_data(x_demean, 1, 0.5, ft)
    [coh, f_IC, _, _, _] = spectral_connectivity(
                                    x_ep, 'coh', sfreq=ft, verbose=False) #, mode='fourier')
    [imcoh, f_IC, _, _, _] = spectral_connectivity(
                                    x_ep, 'imcoh', sfreq=ft, verbose=False) #, mode='fourier')
    index_band = np.where((f_IC >= f_min) & (f_IC <= f_max))[0]
    mean_coh = np.mean(coh[:, :, index_band], axis=2)
    mean_ic = np.mean(abs(imcoh[:, :, index_band]), axis=2)
    
    # 2.5. Save
    
    _aux = {'ft' : ft, 'x' : x, 'A' : A, 'seed' : seed, 
            'mean_coh' : mean_coh, 'mean_ic' : mean_ic}
    aux_f = open(target_file_xint, 'wb')
    pickle.dump(_aux, aux_f, protocol=2)
    aux_f.close()

else:
    
    subject = subjects[subject_id]
    print('Run number = %d - subject %s'%(i_run+1, subject))
    sys.stdout.flush()

# In[]: Step 3. Load
    # 3.1. Forward model
    path_lf = string_lf.format(subject)
    fwd = read_forward_solution(path_lf)
    fwd = convert_forward_solution(fwd, 
                surf_ori=True, force_fixed=True, use_cps=True)
    fwd = pick_types_forward(fwd, meg=sensors_meg, 
                eeg=False, ref_meg=False, exclude='bads')
    nvert = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']
    
    # 3.2. Anatomical regions
    label_lh = read_labels_from_annot(subject=subject, parc=parc, hemi='lh',
                           subjects_dir=subjects_dir)
    label_rh = read_labels_from_annot(subject=subject, parc=parc, hemi='rh', 
                        subjects_dir=subjects_dir)
    label = label_lh + label_rh
    
    # 3.3. Time courses of the interacting sources
    with open(target_file_xint, 'rb') as aux_lf:
        tmp_sign = pickle.load(aux_lf)
    x_int = tmp_sign['x']
    ft = tmp_sign['ft']
    
    T = x_int.shape[1]/ft
    time = np.arange(0, T, 1/ft)
    
    # 3.4. Initialize dictionary where results will be stored  
    ic_fl_mean = dict()
    coh_fl_mean = dict()
    sign_values = dict()
    sign_values_coh = dict()
    
    ic_an_mean = dict()
    coh_an_mean = dict()
    sign_values_an = dict()
    sign_values_an_coh = dict()
    
# In[]: Step 4. Simulate meg signals
    seed_sim = np.random.randint(0, 2**32-1)
    np.random.seed(seed_sim)
    
    L = fwd['sol']['data']
    Ns = L.shape[0]
        
    # 4.1. Brain signals of interest
    n_sint, n_time = x_int.shape
    
    min_ratio_norm = 1/3
    ratio_norm = 0
    while ratio_norm < min_ratio_norm:
        print('run location')
        loc_int = np.random.permutation(nvert)[:n_sint]
        # -- Signal
        y_int_sing = np.zeros([n_sint, Ns, n_time])
        norm_sing = np.zeros([n_sint])
        for ii in range(n_sint):
            Lii = L[:, loc_int[ii]]
            y_int_sing[ii] = Lii[:, np.newaxis]*x_int[ii]
            norm_sing[ii] = np.linalg.norm(y_int_sing[ii], ord='fro')
        norm_sing.sort()
        ratio_norm = norm_sing[0] / norm_sing[1]
    
    y_int = np.sum(y_int_sing, axis=0)
    
    # 4.2. Biological noise (with two different snrs)
    n_sbio = 500

    loc_bio = np.random.permutation(
            np.delete(np.arange(nvert), loc_int))[0:n_sbio]

    x_bio = np.random.randn(n_sbio, n_time)

    y_bio_tmp = np.dot(L[:, loc_bio], x_bio)
    band = 2/ft * np.array([f_min, f_max])
    [b, a] = signal.butter(3, band, btype='bandpass')
    y_bio_tmp_filt = signal.filtfilt(b, a, y_bio_tmp)
    
    for alpha_bio in alpha_bio_tot:   
        y_bio = (alpha_bio * np.linalg.norm(y_int, ord='fro') / 
            np.linalg.norm(y_bio_tmp_filt, ord='fro')) * y_bio_tmp

    # 4.3. Sensor noise
        y_brain = y_int + y_bio
        
        sigma_n = np.mean(np.var(y_brain, axis=1, ddof=1)) / snr
        noise = np.sqrt(sigma_n) * np.random.randn(Ns, time.shape[0])
        C = sigma_n * np.eye(Ns) 
        
        # -- Quick check:
        print('Expected snr = %1.3f - estimated snr = %1.3f'%(snr, 
            np.mean(np.var(y_brain, axis=1, ddof=1) / \
                    np.var(noise, axis=1, ddof=1))))
        
    # 4.4. Overall signal
        y = y_brain + noise
        
# In[]: Step 5. Source estimation (SE)
        for (knn, gamma) in itertools.product(knn_tot, gamma_tot):
            
    # 5.1. Load flame parcellation
            target_file = string_target_file.format(
                                subject, knn, gamma, theta)
            print('Loading %s'%target_file)
            sys.stdout.flush()
            with open(target_file, 'rb') as aux_lf:
                flame_data = pickle.load(aux_lf)
              
    # 5.3. SE
            L_fl = L[:, flame_data['centroids_id']]
            W_fl = f_inverse.compute_inv_op_rank(L_fl, C, lam, 
                                             depth=depth, method=method)
            x_fl = W_fl.dot(y)
        
# In[]: Step 6. Connectivity analysis (CA)
            x_fl_demean = x_fl - np.mean(x_fl, axis=1)[:, None]
            x_fl_ep = f_inverse.window_data(x_fl_demean, 1, 0.5, ft)
            
    # 6.1. Compute connectivity metrics
            [ic_fl, f_IC, _, _, _] = spectral_connectivity(
                                   x_fl_ep, 'imcoh', sfreq=ft, verbose=False)
                                    #fmin = fmin, fmax = fmax, verbose=False)                                    
            band_F = np.where((f_IC >= f_min) & (f_IC <= f_max))[0]
            ic_fl_mean_tmp = np.mean(abs(ic_fl[:, :, band_F]), axis=2)
            
            # -- coh
            [coh_fl, f_IC, _, _, _] = spectral_connectivity(
                                   x_fl_ep, 'coh', sfreq=ft, verbose=False)
                                    #fmin = fmin, fmax = fmax, verbose=False)                                    
            band_F = np.where((f_IC >= f_min) & (f_IC <= f_max))[0]
            coh_fl_mean_tmp = np.mean(abs(coh_fl[:, :, band_F]), axis=2)
            
    # 6.b. Statistical test
            print('Performing statistical test - may require some time')
            sys.stdout.flush()
            
            null_distr = np.zeros(N_surr)
            null_distr[N_surr-1] = np.max(ic_fl_mean_tmp)
            
            null_distr_coh = np.zeros(N_surr)
            null_distr_coh[N_surr-1] = np.max(coh_fl_mean_tmp)
            
            for i_s in range(N_surr-1):
                
                x_surr = f_inverse.generate_surrogate(x_fl_demean, method_surr)
                x_surr_demean = x_surr - np.mean(x_surr, axis=1)[:, None]
                x_surr_ep = f_inverse.window_data(x_surr_demean, 1, 0.5, ft)
                
                [tmp_imcoh, tmp_f_IC, _, _, _] = spectral_connectivity(
                            x_surr_ep, 'imcoh', sfreq=ft, verbose=False, 
                            fmin=f_min, fmax=f_max)
                null_distr[i_s] = np.max(np.mean(abs(tmp_imcoh), axis=2))
                
                [tmp_coh, _, _, _, _] = spectral_connectivity(
                            x_surr_ep, 'coh', sfreq=ft, verbose=False, 
                            fmin=f_min, fmax=f_max) #, mode='fourier')
                null_distr_coh[i_s] = np.max(np.mean(abs(tmp_coh), axis=2))
                
            threshold = np.sort(null_distr)[idx_alpha]
            sign_values_tmp = (ic_fl_mean_tmp > threshold).astype(int)
            
            threshold_coh = np.sort(null_distr_coh)[idx_alpha]
            sign_values_coh_tmp = (coh_fl_mean_tmp > threshold_coh).astype(int)
        
    # 6.e. Store results
            ic_fl_mean['k%d_gamma%1.2f_ab%1.2f'%(knn, gamma, alpha_bio)] = \
                                        ic_fl_mean_tmp
            coh_fl_mean['k%d_gamma%1.2f_ab%1.2f'%(knn, gamma, alpha_bio)] = \
                                        coh_fl_mean_tmp
            sign_values['k%d_gamma%1.2f_ab%1.2f'%(knn, gamma, alpha_bio)] = \
                                        sign_values_tmp
            sign_values_coh['k%d_gamma%1.2f_ab%1.2f'%(knn, gamma, alpha_bio)] = \
                                        sign_values_coh_tmp

        
    # In[]: Step 7: Comparison with anatomical regions
        # 7.a. Find the anatomical regions of the interacting sources
        
        
        # 7.b. SE on full source-space
        W_full = f_inverse.compute_inv_op_rank(L, C, lam, depth=depth, 
                                                method=method, rank=Ns)
        x_full = W_full.dot(y)
        
        # 7.c. Collapsing procedure
        src = fwd['src']
        stc_aux = SourceEstimate(x_full, [src[0]['vertno'], src[1]['vertno']], 
                                         tmin = time[0], 
                                         tstep = time[1]-time[0], 
                                         subject = subject)
        x_an = extract_label_time_course(stc_aux, label, src, mode=method_cp)
        
        # 7.d. CA
        x_an_demean = x_an - np.mean(x_an, axis=1)[:, None]
        x_an_ep = f_inverse.window_data(x_an_demean, 1, 0.5, ft)
        
        [ic_an, f_IC, _, _, _] = spectral_connectivity(
                                    x_an_ep, 'imcoh', sfreq=ft, verbose=False)
        band_F = np.where((f_IC >= f_min) & (f_IC <= f_max))[0]                                
        ic_an_mean_tmp = np.mean(abs(ic_an[:, :, band_F]), axis=2)
        
        [coh_an, f_IC, _, _, _] = spectral_connectivity(
                                    x_an_ep, 'coh', sfreq=ft, verbose=False)
        band_F = np.where((f_IC >= f_min) & (f_IC <= f_max))[0]                                
        coh_an_mean_tmp = np.mean(abs(coh_an[:, :, band_F]), axis=2)
        
        ic_an_mean['ab_%1.2f'%alpha_bio] = ic_an_mean_tmp
        coh_an_mean['ab_%1.2f'%alpha_bio] = coh_an_mean_tmp
        
        # 7.e. Statistical test
        print('Performing statistical test (Anatomical ROI) - may require some time')
        sys.stdout.flush()
    
        null_distr_an = np.zeros(N_surr)
        null_distr_an[N_surr-1] = np.max(ic_an_mean_tmp)
        
        null_distr_an_coh = np.zeros(N_surr)
        null_distr_an_coh[N_surr-1] = np.max(coh_an_mean_tmp)
        
        for i_s in range(N_surr-1):
            
            x_surr = f_inverse.generate_surrogate(x_an_demean, method_surr)
            x_surr_demean = x_surr - np.mean(x_surr, axis=1)[:, None]
            x_surr_ep = f_inverse.window_data(x_surr_demean, 1, 0.5, ft)
            
            [tmp_imcoh, tmp_f_IC, _, _, _] = spectral_connectivity(
                                    x_surr_ep, 'imcoh', sfreq=ft, verbose=False,
                                    fmin=f_min, fmax=f_max) #, mode='fourier')
            null_distr_an[i_s] = np.max(np.mean(abs(tmp_imcoh), axis=2))
            
            [tmp_coh, _, _, _, _] = spectral_connectivity(
                                    x_surr_ep, 'coh', sfreq=ft, verbose=False,
                                    fmin=f_min, fmax=f_max) #, mode='fourier')
            null_distr_an_coh[i_s] = np.max(np.mean(abs(tmp_coh), axis=2))
            
        threshold_an = np.sort(null_distr_an)[idx_alpha]    
        sign_values_an['ab_%1.2f'%alpha_bio] = (ic_an_mean_tmp > threshold_an).astype(int)
        
        threshold_an_coh = np.sort(null_distr_an_coh)[idx_alpha]
        sign_values_an_coh['ab_%1.2f'%alpha_bio] = (coh_an_mean_tmp > threshold_an_coh).astype(int)
        
    # In[]: Step 8: Save results
    
    # 8.4. Check and memorize the regions of the interacting sources
    # - flame
    fl_regions = dict()
    for (knn, gamma) in itertools.product(knn_tot, gamma_tot):
        target_file = string_target_file.format(subject, knn, gamma, theta)
        with open(target_file, 'rb') as aux_lf:
            flame_data = pickle.load(aux_lf)
        memberships = flame.membership2vector(
                flame_data['parcel'][:flame_data['centroids']], nvert)
        fl_regions['k%d_gamma%1.2f'%(knn, gamma)] = memberships[loc_int]
    
    # - anatomical
    proj_anrois = flame.labels_to_array(label, fwd['src'])
    if proj_anrois['outliers'] > 0:
        aux_num_roi = len(proj_anrois['parcel']) - 1
    else:
        aux_num_roi = len(proj_anrois['parcel'])
    memberships_an = flame.membership2vector(
                proj_anrois['parcel'][:aux_num_roi], nvert)
    an_regions = memberships_an[loc_int]
    
    # 8.b. Save
    dipoles = {'loc_int' : loc_int, 
	       'fl_regions' : fl_regions, 
	       'an_regions' : an_regions, 
           'seed' : seed_sim}

    conn_est = {'ic_fl_mean' : ic_fl_mean,
                'coh_fl_mean' : coh_fl_mean,
               'ic_an_mean' : ic_an_mean,
               'coh_an_mean' : coh_an_mean, 
               'sign_values' : sign_values, 
               'sign_values_coh' : sign_values_coh, 
               'sign_values_an' : sign_values_an, 
               'sign_values_an_coh' : sign_values_an_coh}
    
    print('Saving...')
    sys.stdout.flush()
    
    _aux = {'dipoles' : dipoles, 'conn_est': conn_est}
    path_res = string_results.format(subject, i_run+1, subject)
    aux_f = open(path_res, 'wb')
    pickle.dump(_aux, aux_f, protocol=2)
    aux_f.close()
