# %%
"""
Analyze experimental MEG data concerning auditory evoked field
==============================================================
"""

import os.path as op
import numpy as np
import pickle

import matplotlib.pyplot as plt

from mne import (read_epochs, read_forward_solution, read_labels_from_annot, 
                 convert_forward_solution, pick_types, pick_types_forward,
                 compute_covariance, SourceEstimate, extract_label_time_course)
from mne.viz import plot_source_estimates, get_brain_class

from sklearn.metrics import roc_curve, auc

import megicparc
from megicparc.utils import read_dipole_locations, compute_inv_op_rank
from megicparc.viz import plot_flame_labels, plot_flame_centroids

target = '../data'
path_fig = op.join(target, 'figures')
subjects_dir = op.join(target, 'subjects_flame')
folder_dipfit = './data'
string_dipfit = op.join(target, 'xfit_dipoles', '{:s}.dip')

# In[]: Step 7. Plot
plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{color}')
plt.rc('font', family='serif', size=22)
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['errorbar.capsize'] = 3.5
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams['patch.force_edgecolor'] = True

stim = 'audL'
parc = 'aparc'
sensors_meg = 'grad'

depth = None
method = 'dSPM'

threshs = np.arange(0, 1.05, 0.05)

isubs = np.array([1, 2, 4, 6])
#isubs = np.array([2])

knn = 30
gamma = 0.4
theta = 0.05

# Time-interval used to estimate the snr
tmin_snr = 0.0
tmax_snr = 0.15

# Time-interval used to compute peaks of source estimates
tmin_res = 0
tmax_res = 0.15

folder_fl = op.join(target, 'results', './parcellations')
string_epo = op.join(target, 'meg_auditory', 'epochs_{:s}_{:s}-epo.fif')
string_lf = op.join(target, 'fwd_models', 'original_fwd','{:s}_meg_single_layer-fwd.fif')
string_parc_file = op.join(folder_fl,
                        '{:s}_flame_grad_k{:d}_gamma{:1.2f}_theta{:1.2f}.pkl')

# In[] Initialization
num_roi = np.zeros(np.size(isubs), dtype=np.int32)
euclerr_full_lh = np.zeros(np.size(isubs))
euclerr_full_rh = np.zeros(np.size(isubs))
euclerr_fl_lh = np.zeros(np.size(isubs))
euclerr_fl_rh = np.zeros(np.size(isubs))
euclerr_flreg_lh = np.zeros(np.size(isubs))
euclerr_flreg_rh = np.zeros(np.size(isubs))

fp_rate_mean = np.zeros((np.size(isubs), np.size(threshs)))
fp_rate_pca = np.zeros((np.size(isubs), np.size(threshs)))
fp_rate_fl = np.zeros((np.size(isubs), np.size(threshs)))

tp_rate_mean = np.zeros((np.size(isubs), np.size(threshs)))
tp_rate_pca = np.zeros((np.size(isubs), np.size(threshs)))
tp_rate_fl = np.zeros((np.size(isubs), np.size(threshs)))

auc_mean = np.zeros((np.size(isubs)))
auc_pca = np.zeros((np.size(isubs)))
auc_fl = np.zeros((np.size(isubs)))

fp_rate_mean_rh = np.zeros((np.size(isubs), np.size(threshs)))
fp_rate_pca_rh = np.zeros((np.size(isubs), np.size(threshs)))
fp_rate_fl_rh = np.zeros((np.size(isubs), np.size(threshs)))

tp_rate_mean_rh = np.zeros((np.size(isubs), np.size(threshs)))
tp_rate_pca_rh = np.zeros((np.size(isubs), np.size(threshs)))
tp_rate_fl_rh = np.zeros((np.size(isubs), np.size(threshs)))

auc_mean_rh = np.zeros((np.size(isubs)))
auc_pca_rh = np.zeros((np.size(isubs)))
auc_fl_rh = np.zeros((np.size(isubs)))

fl_thrs = dict()
fl_fpr = dict()
fl_tpr = dict()
mean_thrs = dict()
mean_fpr = dict()
mean_tpr = dict()
pca_thrs = dict()
pca_fpr = dict()
pca_tpr = dict()

fl_auc = np.zeros((np.size(isubs)))
mean_auc = np.zeros((np.size(isubs)))
pca_auc = np.zeros((np.size(isubs)))

fl_thrs_rh = dict()
fl_fpr_rh = dict()
fl_tpr_rh = dict()
mean_thrs_rh = dict()
mean_fpr_rh = dict()
mean_tpr_rh = dict()
pca_thrs_rh = dict()
pca_fpr_rh = dict()
pca_tpr_rh = dict()

fl_auc_rh = np.zeros((np.size(isubs)))
mean_auc_rh = np.zeros((np.size(isubs)))
pca_auc_rh = np.zeros((np.size(isubs)))

# In[]: Step 2. Load
for idx_s, isub in enumerate(isubs):
    
    subject_name = 'sub0' + str(isub)
    subject = 'k'+ str(isub) + '_T1'
    
#   2.1. Preprocessed epoched data
    path_epo = string_epo.format(subject_name, stim)
    epochs = read_epochs(path_epo)
    epochs.apply_baseline((None, 0)) 
    evoked = epochs.average()

#   2.2. Forward model
    path_lf = string_lf.format(subject)
    fwd = read_forward_solution(path_lf)
    fwd = convert_forward_solution(fwd, 
            surf_ori=True, force_fixed=True, use_cps=True)
#   --> Pick only selected channels
    picks_inv = pick_types(evoked.info, meg=sensors_meg, exclude='bads') 
    y_ev = evoked.data[picks_inv, :]        
    fwd = pick_types_forward(fwd, meg=sensors_meg, eeg=False, exclude='bads')

#   2.3. Anatomcal regions
    label_lh = read_labels_from_annot(subject=subject, parc=parc, hemi='lh',
                           subjects_dir=subjects_dir)
    label_rh = read_labels_from_annot(subject=subject, parc=parc, hemi='rh', 
                        subjects_dir=subjects_dir)
    label = label_lh + label_rh

#   2.3. Flame parcellation
    target_file = string_parc_file.format(subject, knn, gamma, theta)
    print('Loading %s'%target_file)
    with open(target_file, 'rb') as aux_lf:
        flame_data = pickle.load(aux_lf)
        
    num_roi[idx_s] = flame_data['centroids']

#   2.4. Reference dipole location
    path_dipfit = string_dipfit.format(subject)
    dipoles_loc = read_dipole_locations(path_dipfit)
    dipoles_loc = dipoles_loc[np.argsort(dipoles_loc[:, 0])]  # Sort left-right hemi

# In[]: Step 3. Estimated noise covariance and related stuff
#   3.1. Estimate C
    noise_cov = compute_covariance(
                epochs, tmax=0., method=['empirical'])
    C = noise_cov.data[np.ix_(picks_inv, picks_inv)] / len(epochs)

#   3.2. Estimate rank
    _, S, V = np.linalg.svd(C)
    log_ratio_s = np.log(S[0:-1]) - np.log(S[1:])
    rank = np.argmax(log_ratio_s)
    rank = rank + 1
    
#   3.3. Estimate snr and regularization parameter
    ssv = np.arange(0, rank)
    W = np.dot(np.diag(1/np.sqrt(S[ssv])), V[ssv])
    yW = W.dot(y_ev)
    est_snr = np.sum(yW**2, axis=0) / rank       
    mean_snr = np.mean(est_snr[
            (evoked.times>=tmin_snr) & (evoked.times <=tmax_snr)])
    lam = 1/mean_snr
    
    print('***************************************')
    print('Estimated rank %2d' %rank)
    print('Estimated snr %2.2f' %mean_snr)
    print('Estimated lam %2.2f' %lam)

# In[]: Step 4. Source estimation on full source-space
    L = fwd['sol']['data']
    W_inv = compute_inv_op_rank(L, C, lam, 
                    depth=depth, method=method, rank=rank)
    x_full = W_inv.dot(y_ev)
    
    #   - collapse activity on anatomical regions
    stc_aux = SourceEstimate(x_full, 
            [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']], 
            tmin = evoked.times[0], tstep = evoked.times[1]-evoked.times[0], 
            subject = subject)
    x_an_mean = extract_label_time_course(stc_aux, label, 
                            fwd['src'], mode='mean_flip')

    x_an_pca = extract_label_time_course(stc_aux, label, 
                            fwd['src'], mode='pca_flip')
    
# In[]: Step 5. Source estimate on reduced source-space
    L_red = fwd['sol']['data'][:, flame_data['centroids_id']]
    W_red = compute_inv_op_rank(L_red, C, lam, 
                    depth=depth, method=method, rank=rank)
    x_fl = W_red.dot(y_ev)

# In[]: Step 6. Evaluation criteria:
    
    #   6.1. Compute peaks of the estimated sources.
    times = evoked.times
    aux_t = np.where((times > tmin_res) & (times < tmax_res))[0]
    
    vertices_loc = np.concatenate((fwd['src'][0]['rr'][fwd['src'][0]['vertno']], 
                                  fwd['src'][1]['rr'][fwd['src'][1]['vertno']]))
    nv_lh = fwd['src'][0]['nuse']
    nf_lh = np.where(flame_data['centroids_id'] < nv_lh)[0].shape[0]
    na_lh = len(label_lh)

    #   - full source-space
    t_peak_lh = aux_t[np.argmax(np.max(abs(x_full[0:nv_lh, aux_t]), axis=0))]
    t_peak_rh = aux_t[np.argmax(np.max(abs(x_full[nv_lh:, aux_t]), axis=0))]
    
    peak_full_lh = np.argmax(abs(x_full[0:nv_lh, t_peak_lh]), axis=0)
    peak_full_rh = np.argmax(abs(x_full[nv_lh:, t_peak_rh]), axis=0) + nv_lh
    
    #   - reduced source-space
    t_peak_lh_fl = aux_t[np.argmax(np.max(abs(x_fl[0:nf_lh, aux_t]), axis=0))]
    t_peak_rh_fl = aux_t[np.argmax(np.max(abs(x_fl[nf_lh:, aux_t]), axis=0))]
    peak_fl_lh = np.argmax(abs(x_fl[0:nf_lh, t_peak_lh_fl]), axis=0)
    peak_fl_rh = np.argmax(abs(x_fl[nf_lh:, t_peak_rh_fl]), axis=0) + nf_lh
    
    #   6.2. Compute localization error with respect to reference fitted dipole
    
    #   - full source-space
    euclerr_full_lh[idx_s] = np.linalg.norm(dipoles_loc[0] - \
                        vertices_loc[peak_full_lh])
    euclerr_full_rh[idx_s] = np.linalg.norm(dipoles_loc[1] - \
                        vertices_loc[peak_full_rh])
    
    #   - reduced source-space
    euclerr_fl_lh[idx_s] = np.linalg.norm(dipoles_loc[0] - \
                        vertices_loc[flame_data['centroids_id'][peak_fl_lh]])
    euclerr_fl_rh[idx_s] = np.linalg.norm(dipoles_loc[1] - \
                        vertices_loc[flame_data['centroids_id'][peak_fl_rh]])
    
    euclerr_flreg_lh[idx_s] = np.min(np.linalg.norm(dipoles_loc[0] - \
                    vertices_loc[flame_data['parcel'][peak_fl_lh]], axis=1))
    euclerr_flreg_rh[idx_s] = np.min(np.linalg.norm(dipoles_loc[1] - \
                    vertices_loc[flame_data['parcel'][peak_fl_rh]], axis=1))
    
    #   6.3. Evaluate the specificity of the proposed approach
    
    #   6.3.1. Check to which (anatomical/flame) region that contain the peak
    #           on full source-space, left hemi
    n_vert = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']
    proj_anrois = megicparc.labels_to_array(label, fwd['src'])
    true_memb_an = megicparc.membership2vector(
            proj_anrois['parcel'][:len(label)], n_vert)
    tp_an = true_memb_an[peak_full_lh]
    true_memb_fl = megicparc.membership2vector(
            flame_data['parcel'][:flame_data['centroids']], n_vert)
    tp_fl = true_memb_fl[peak_full_lh]
    
    #   6.3.2. Compute normalized activity at the time point of the peaks
    
    #   - DK flipped mean
    t_peak_lh_mean = aux_t[np.argmax(np.max(
            abs(x_an_mean[0:na_lh, aux_t]), axis=0))]
    aux_ = abs(x_an_mean[0:na_lh, t_peak_lh_mean])
    norm_act_mean = aux_ / np.max(aux_)
    del aux_
    
    #   - DK PCA
    t_peak_lh_pca = aux_t[np.argmax(np.max(
            abs(x_an_pca[0:na_lh, aux_t]), axis=0))]
    aux_ = abs(x_an_pca[0:na_lh, t_peak_lh_pca])
    norm_act_pca = aux_ / np.max(aux_)
    del aux_
    
    #   - flame
    aux_ = abs(x_fl[0:nf_lh, t_peak_lh_fl])
    norm_act_fl = aux_ / np.max(aux_)
    del aux_
    
    #  6.3.3. compute roc curves (Left hemi)
    aux_true = np.zeros(na_lh)
    aux_true[tp_an] = 1
    mean_fpr[subject_name], mean_tpr[subject_name], mean_thrs[subject_name] = \
        roc_curve(aux_true, norm_act_mean, pos_label=1, drop_intermediate=False)
    mean_auc[idx_s] = auc(mean_fpr[subject_name], mean_tpr[subject_name])
    pca_fpr[subject_name], pca_tpr[subject_name], pca_thrs[subject_name] = \
        roc_curve(aux_true, norm_act_pca, pos_label=1, drop_intermediate=False)
    pca_auc[idx_s] = auc(pca_fpr[subject_name], pca_tpr[subject_name])
        
    aux_true = np.zeros(nf_lh)
    aux_true[tp_fl] = 1
    fl_fpr[subject_name], fl_tpr[subject_name], fl_thrs[subject_name] = \
        roc_curve(aux_true, norm_act_fl, pos_label=1, drop_intermediate=False)
    fl_auc[idx_s] = auc(fl_fpr[subject_name], fl_tpr[subject_name])

    #   6.3.3. Compute false positive rates
    for idx_t, th in enumerate(threshs):
        
    #   - DK flipped mean
        sign_values = np.where(norm_act_mean>=th)[0]
        positive = 0
        if tp_an in sign_values:
            positive = 1
        tp_rate_mean[idx_s, idx_t] = positive
        fp_rate_mean[idx_s, idx_t] = (np.size(sign_values) - positive) / (na_lh-1)
        del sign_values, positive
    
    #   - DK PCA
        sign_values = np.where(norm_act_pca>=th)[0]
        positive = 0
        if tp_an in sign_values:
            positive = 1
        tp_rate_pca[idx_s, idx_t] = positive
        fp_rate_pca[idx_s, idx_t] = (np.size(sign_values) - positive) / (na_lh-1)
        del sign_values, positive
    
    #   - flame
        sign_values = np.where(norm_act_fl>=th)[0]
        positive = 0
        if tp_fl in sign_values:
            positive = 1
        tp_rate_fl[idx_s, idx_t] = positive
        fp_rate_fl[idx_s, idx_t] = (np.size(sign_values) - positive) / (nf_lh-1)
        del sign_values, positive
    
    auc_mean[idx_s] = auc(fp_rate_mean[idx_s], tp_rate_mean[idx_s])
    auc_pca[idx_s] = auc(fp_rate_pca[idx_s], tp_rate_pca[idx_s])
    auc_fl[idx_s] = auc(fp_rate_fl[idx_s], tp_rate_fl[idx_s])
        
    # 6.4. Check what happen in the right hemi
    # 6.4.1.
    tp_an_rh = true_memb_an[peak_full_rh]
    tp_fl_rh = true_memb_fl[peak_full_rh]
    
    # 6.4.2.
    #   - DK flipped mean
    t_peak_rh_mean = aux_t[np.argmax(np.max(
            abs(x_an_mean[na_lh:, aux_t]), axis=0))]
    aux_ = abs(x_an_mean[na_lh:, t_peak_rh_mean])
    norm_act_mean = aux_ / np.max(aux_)
    del aux_
    
    #   - DK PCA
    t_peak_rh_pca = aux_t[np.argmax(np.max(
            abs(x_an_pca[na_lh:, aux_t]), axis=0))]
    aux_ = abs(x_an_pca[na_lh:, t_peak_rh_pca])
    norm_act_pca = aux_ / np.max(aux_)
    del aux_
    
    #   - flame
    aux_ = abs(x_fl[nf_lh:, t_peak_rh_fl])
    norm_act_fl = aux_ / np.max(aux_)
    del aux_
    
    #   6.3.4. Compute false positive rates
    na_rh = len(label_rh)
    nf_rh = flame_data['centroids'] - nf_lh

    # Compute roc curves (Right hemi)
    aux_true = np.zeros(na_rh)
    aux_true[tp_an_rh-na_lh] = 1
    mean_fpr_rh[subject_name], mean_tpr_rh[subject_name], mean_thrs_rh[subject_name] = \
        roc_curve(aux_true, norm_act_mean, pos_label=1, drop_intermediate=False)
    mean_auc_rh[idx_s] = auc(mean_fpr_rh[subject_name], mean_tpr_rh[subject_name])
    pca_fpr_rh[subject_name], pca_tpr_rh[subject_name], pca_thrs_rh[subject_name] = \
        roc_curve(aux_true, norm_act_pca, pos_label=1, drop_intermediate=False)
    pca_auc_rh[idx_s] = auc(pca_fpr_rh[subject_name], pca_tpr_rh[subject_name])
        
    aux_true = np.zeros(nf_rh)
    aux_true[tp_fl-nf_lh] = 1
    fl_fpr_rh[subject_name], fl_tpr_rh[subject_name], fl_thrs_rh[subject_name] = \
        roc_curve(aux_true, norm_act_fl, pos_label=1, drop_intermediate=False)
    fl_auc_rh[idx_s] = auc(fl_fpr_rh[subject_name], fl_tpr_rh[subject_name])
    
    for idx_t, th in enumerate(threshs):
        
    #   - DK flipped mean
        sign_values = np.where(norm_act_mean>=th)[0]
        positive = 0
        if tp_an_rh - na_lh in sign_values:
            positive = 1
        tp_rate_mean_rh[idx_s, idx_t] = positive
        fp_rate_mean_rh[idx_s, idx_t] = (np.size(sign_values) - positive) / (na_rh-1)
        del sign_values, positive
    
    #   - DK PCA
        sign_values = np.where(norm_act_pca>=th)[0]
        positive = 0
        if tp_an_rh - na_lh in sign_values:
            positive = 1
        tp_rate_pca_rh[idx_s, idx_t] = positive
        fp_rate_pca_rh[idx_s, idx_t] = (np.size(sign_values) - positive) / (na_rh-1)
        del sign_values, positive
    
    #   - flame
        sign_values = np.where(norm_act_fl>=th)[0]
        positive = 0
        if tp_fl_rh - nf_lh in sign_values:
            positive = 1
        tp_rate_fl_rh[idx_s, idx_t] = positive
        fp_rate_fl_rh[idx_s, idx_t] = (np.size(sign_values) - positive) / (nf_rh-1)
        del sign_values, positive

    auc_mean_rh[idx_s] = auc(fp_rate_mean_rh[idx_s], tp_rate_mean_rh[idx_s])
    auc_pca_rh[idx_s] = auc(fp_rate_pca_rh[idx_s], tp_rate_pca_rh[idx_s])
    auc_fl_rh[idx_s] = auc(fp_rate_fl_rh[idx_s], tp_rate_fl_rh[idx_s])
    
    # Individual plots
    # In[]: Additional P1. Data and estimated snr
    f_d, ax_d = plt.subplots(3)

    ax_d[0].plot(evoked.times, y_ev.T)
    ax_d[0].set_ylabel('y(t)')

    ax_d[1].plot(evoked.times, yW.T)
    ax_d[1].plot(evoked.times, 2*np.ones(evoked.times.shape), 
          'k',  linewidth=2)
    ax_d[1].plot(evoked.times, -2*np.ones(evoked.times.shape), 
          'k', linewidth=2)
    ax_d[1].set_ylabel('y(t) - white')

    ax_d[2].plot(evoked.times, est_snr)
    ax_d[2].plot(tmin_snr*np.ones(2), [0, np.max(est_snr)], 'r')
    ax_d[2].plot(tmax_snr*np.ones(2), [0, np.max(est_snr)], 'r')
    ax_d[2].set_ylabel('SNR(t)')
    ax_d[2].set_xlabel('t [ms]')
    ax_d[2].set_title('SNR = %2.2f - lam = %2.2f' %(mean_snr, lam))
    
# In[]: P2. Cancellation effects of collapsing procedures
    f_ce, ax_ce = plt.subplots(3, 2)
    ax_ce[0, 0].plot(evoked.times, abs(x_full[0:nv_lh].T))
    ax_ce[0, 0].set_ylabel(r'dSPM value $\mathcal{V}$', 
                            fontsize=24)
    ax_ce[0, 0].set_ylim([0, np.max(abs(x_full))])
    ax_ce[0, 0].set_xlim([0, 0.4])
    ax_ce[0, 0].set_xticklabels([])
    ax_ce[0, 0].set_title('Left hemisphere')
    
    ax_ce[0, 1].plot(evoked.times, abs(x_full[nv_lh:].T))
    ax_ce[0, 1].set_ylim([0, np.max(abs(x_full))])
    ax_ce[0, 1].set_xlim([0, 0.4])
    ax_ce[0, 1].set_xticklabels([])
    ax_ce[0, 1].set_title('Right hemisphere')
        
    ax_ce[1, 0].plot(evoked.times, abs(x_an_mean[0:na_lh].T))
    ax_ce[1, 0].set_ylim([0, np.max(abs(x_full))])
    ax_ce[1, 0].set_ylabel(r'dSPM value \\ DK atlas (mean)', 
                            fontsize=24)
    ax_ce[1, 0].set_xlim([0, 0.4])
    ax_ce[1, 0].set_xticklabels([])
        
    ax_ce[1, 1].plot(evoked.times, abs(x_an_mean[na_lh:].T))
    ax_ce[1, 1].set_ylim([0, np.max(abs(x_full))])
    ax_ce[1, 1].set_xlim([0, 0.4])
    ax_ce[1, 1].set_xticklabels([])
    
    ax_ce[2, 0].plot(evoked.times, abs(x_an_pca[0:na_lh].T))
    ax_ce[2, 0].set_ylim([0, np.max(abs(x_full))])
    ax_ce[2, 0].set_ylabel(r'dSPM value \\ DK atlas (PCA)',
                            fontsize=24)
    ax_ce[2, 0].set_xlim([0, 0.4])
    ax_ce[2, 0].set_xlabel('Time [ms]')
        
    ax_ce[2, 1].plot(evoked.times, abs(x_an_pca[na_lh:].T))
    ax_ce[2, 1].set_ylim([0, np.max(abs(x_full))])
    ax_ce[2, 1].set_xlim([0, 0.4])
    ax_ce[2, 1].set_xlabel('Time [ms]')
    
    f_ce.set_size_inches(12, 8.5)
    plt.tight_layout(pad=1.5)
    f_ce.savefig(op.join(path_fig, '%s_%s_cancellation_effects.png'%(
                                         subject, stim)))
    
# In[]: P3. Source estimated on full source-space
    vertices_plot = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    flame_labels = megicparc.store_flame_labels(
                            flame_data, fwd['src'], subject)

    Brain = get_brain_class()
    x_full_norm = np.zeros(x_full.shape[0])
    x_full_norm[0:nv_lh] = abs(x_full[0:nv_lh, t_peak_lh]) \
                                / np.max(abs(x_full[0:nv_lh, t_peak_lh]))
    x_full_norm[nv_lh:] = abs(x_full[nv_lh:, t_peak_rh]) \
                                / np.max(abs(x_full[nv_lh:, t_peak_rh]))

    stc_full = SourceEstimate(x_full_norm[:, np.newaxis], vertices_plot, 
                              tmin = 0, tstep = 1, subject=subject)

    brain = plot_source_estimates(stc_full, subject, surface='inflated', 
                hemi='both', subjects_dir=subjects_dir, background='white', 
                foreground='black', time_label=None, time_viewer=False,
                clim={'kind' : 'value', 'lims' : [0.1, 0.5, 1]}, size=(650, 650))
    if subject == 'k2_T1':
        brain.show_view(azimuth=0, elevation=90,distance=600)
    else:
        brain.show_view(azimuth=0, elevation=90)
    brain.add_text(0.95, 0.13, 'Time = %0.3f s'%times[t_peak_rh], 
                   font_size=20, justification='right', color='black')
    brain.save_image(op.join(path_fig,  
        '%s_%s_k%d_gamma%1.2f_full_rh.png'%(subject, stim, knn, gamma)))
    
    brain2 = plot_source_estimates(stc_full, subject, surface='inflated', 
                hemi='both', subjects_dir=subjects_dir, background='white', 
                foreground='black', time_label=None, time_viewer=False,
                clim={'kind' : 'value', 'lims' : [0.1, 0.5, 1]}, size=(650, 650))
    if subject == 'k2_T1':
        brain2.show_view(azimuth=180, elevation=90,distance=600)
    else:
        brain2.show_view(azimuth=180, elevation=90,distance=600) 
    brain2.add_text(0.05, 0.13, 'Time = %0.3f s'%times[t_peak_lh], 
                   font_size=16, justification='left', color='black')
    brain2.save_image(op.join(path_fig,  
        '%s_%s_k%d_gamma%1.2f_full_lh.png'%(subject, stim, knn, gamma)))
    
    # In[]: P4. Source estimation on the reduces source-space
    idx_fl = [peak_fl_lh+1, peak_fl_rh+1]
    brain_fl = Brain(subject, hemi='both', surf='inflated', background='white',
                subjects_dir=subjects_dir, alpha=1, size=(650, 650))
    plot_flame_labels(idx_fl, flame_labels, fwd['src'], 
                subject, subjects_dir, color = [1, 0.64, 0.], brain=brain_fl)
    plot_flame_centroids(flame_data, fwd['src'], subject, 
                subjects_dir, brain=brain_fl, scale_factor=0.65)
    if subject == 'k2_T1':
        brain_fl.show_view(azimuth=0, elevation=90,distance=600)
    else:
        brain_fl.show_view(azimuth=0, elevation=90)
    brain_fl.save_image(op.join(path_fig,  
        '%s_%s_k%d_gamma%1.2f_fl_rh.png'%(subject, stim, knn, gamma)))
    if subject == 'k2_T1':
        brain_fl.show_view(azimuth=180, elevation=90,distance=600)
    else:
        brain_fl.show_view(azimuth=180, elevation=90)
    brain_fl.save_image(op.join(path_fig,  
        '%s_%s_k%d_gamma%1.2f_fl_lh.png'%(subject, stim, knn, gamma)))
        


# %%
# Table of the localization errors
tab_num_reg = 'Number regions: \n'
for idx_s, isub in enumerate(isubs):
    tab_num_reg += 'sub0%d %d \n'%(idx_s+1, num_roi[idx_s])

euclerr_full_lh *= 10**3 # convert from m to mm
euclerr_full_rh *= 10**3
euclerr_fl_lh *= 10**3
euclerr_fl_rh *= 10**3
euclerr_flreg_lh *= 10**3
euclerr_flreg_rh *= 10**3

tab_num_reg = 'Number regions: \n'
for idx_s, isub in enumerate(isubs):
    tab_num_reg += 'sub0%d %d \n'%(idx_s+1, num_roi[idx_s])
    
tab_eucl_err = '        Left hemi        |        Right hemi \n'
tab_eucl_err += '  full sp  red sp  region | full sp  red sp  region \n'
for idx_s, isub in enumerate(isubs):
    tab_eucl_err += 'sub0%d %1.2f %1.2f %1.2f | %1.2f %1.2f %1.2f \n'%(
            idx_s+1, 
            euclerr_full_lh[idx_s], euclerr_fl_lh[idx_s], euclerr_flreg_lh[idx_s], 
            euclerr_full_rh[idx_s], euclerr_fl_rh[idx_s], euclerr_flreg_rh[idx_s])

print(tab_num_reg)
print(tab_eucl_err)

# %%
# Plot 1: False positive rate and AUC
fp_rate_mean_ave = np.mean(fp_rate_mean, axis=0)
fp_rate_mean_sem = np.std(fp_rate_mean, axis=0) / np.sqrt(np.size(isubs))

fp_rate_pca_ave = np.mean(fp_rate_pca, axis=0)
fp_rate_pca_sem = np.std(fp_rate_pca, axis=0) / np.sqrt(np.size(isubs))

fp_rate_fl_ave = np.mean(fp_rate_fl, axis=0)
fp_rate_fl_sem = np.std(fp_rate_fl, axis=0) / np.sqrt(np.size(isubs))

auc_mean_ave = np.mean(auc_mean)
auc_mean_sem = np.std(auc_mean) / np.sqrt(np.size(isubs))
auc_pca_ave = np.mean(auc_pca)
auc_pca_sem = np.std(auc_pca) / np.sqrt(np.size(isubs))
auc_fl_ave = np.mean(auc_fl)
auc_fl_sem = np.std(auc_fl) / np.sqrt(np.size(isubs))

f_fpr = plt.figure()
plt.errorbar(threshs, fp_rate_fl_ave, yerr=fp_rate_fl_sem, 
             label='MEG-informed ({:1.2f} $\pm$ {:1.3f})'.format(auc_fl_ave, auc_fl_sem), 
             linewidth=3, color='k')
plt.errorbar(threshs, fp_rate_mean_ave, yerr=fp_rate_mean_sem, 
             label='DK  mean $\quad \quad $ ({:1.2f} $\pm$ {:1.3f})'.format(auc_mean_ave, auc_mean_sem),
             linewidth=3, color='r')
plt.errorbar(threshs, fp_rate_pca_ave, yerr=fp_rate_pca_sem, 
             label='DK PCA $\quad \quad \, $  ({:1.2f} $\pm$ {:1.3f})'.format(auc_pca_ave, auc_pca_sem),
             linewidth=3, color='g')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.legend()
plt.ylabel('False positive rate', fontsize=30)
plt.xlabel('Relative detection threshold', fontsize=30)
plt.grid()
plt.show()

f_fpr.set_size_inches(9, 6.8)
plt.tight_layout(pad=1)

f_fpr.savefig(op.join(path_fig, '%s_specificity.png'%stim))

# # In[]: Right hemi
# fp_rate_mean_ave_rh = np.mean(fp_rate_mean_rh, axis=0)
# fp_rate_mean_sem_rh = np.std(fp_rate_mean_rh, axis=0) / np.sqrt(np.size(isubs))

# fp_rate_pca_ave_rh = np.mean(fp_rate_pca_rh, axis=0)
# fp_rate_pca_sem_rh = np.std(fp_rate_pca_rh, axis=0) / np.sqrt(np.size(isubs))

# fp_rate_fl_ave_rh = np.mean(fp_rate_fl_rh, axis=0)
# fp_rate_fl_sem_rh = np.std(fp_rate_fl_rh, axis=0) / np.sqrt(np.size(isubs))

# auc_mean_ave_rh = np.mean(auc_mean_rh)
# auc_mean_sem_rh = np.std(auc_mean_rh) / np.sqrt(np.size(isubs))
# auc_pca_ave_rh = np.mean(auc_pca_rh)
# auc_pca_sem_rh = np.std(auc_pca_rh) / np.sqrt(np.size(isubs))
# auc_fl_ave_rh = np.mean(auc_fl_rh)
# auc_fl_sem_rh = np.std(auc_fl_rh) / np.sqrt(np.size(isubs))

# f_fpr_rh = plt.figure()
# plt.errorbar(threshs, fp_rate_fl_ave_rh, yerr=fp_rate_fl_sem, 
#              label='MEG-informed ({:1.2f} $\pm$ {:1.3f})'.format(auc_fl_ave_rh, auc_fl_sem_rh), 
#              linewidth=3, color='k')
# plt.errorbar(threshs, fp_rate_mean_ave_rh, yerr=fp_rate_mean_sem, 
#              label='DK  mean $\quad \quad $ ({:1.2f} $\pm$ {:1.3f})'.format(auc_mean_ave_rh, auc_mean_sem_rh),
#              linewidth=3, color='r')
# plt.errorbar(threshs, fp_rate_pca_ave_rh, yerr=fp_rate_pca_sem, 
#              label='DK PCA $\quad \quad \, $  ({:1.2f} $\pm$ {:1.3f})'.format(auc_pca_ave_rh, auc_pca_sem_rh),
#              linewidth=3, color='g')
# plt.ylim(0, 1)
# plt.xlim(0, 1)
# plt.legend()
# plt.ylabel('False positive rate')
# plt.xlabel('Relative detection threshold')
# plt.grid()
# plt.show()

# f_fpr_rh.savefig(op.join(path_fig, '%s_specificity_right.png'%stim))



# %%
