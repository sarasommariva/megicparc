# %%
"""
Functional properties of the MEG-informed parcellation
======================================================
"""

import os.path as op
import numpy as np
from scipy import stats
import pickle

from mne import (read_forward_solution, pick_types_forward, 
                 convert_forward_solution, read_labels_from_annot,
                 SourceEstimate)
from mne.viz import plot_brain_colorbar

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

import megicparc
from megicparc.utils import compute_inv_op_rank, collapse_RM
from megicparc.viz import plot_flame_centroids

# %%
# In[]: Step 1. Define general parameters
target = '../data'
path_fig = op.join(target, 'figures')
subjects_dir = op.join(target, 'subjects_flame')
subjects = ['k1_T1', 'k2_T1', 'k4_T1', 'k6_T1', 'k7_T1',
           'CC110045', 'CC110182', 'CC110174', 'CC110126', 'CC110056']
sensors_meg = 'grad' 
string_lf = op.join(target, 'fwd_models', 'original_fwd',
                  '{:s}_meg_single_layer-fwd.fif')

gamma_tot = np.arange(0, 1.01, 0.2)
knn_tot = [10, 20, 30, 40]
theta = 0.05
parc = 'aparc'

snr = 3
lam = 1. / snr
method = 'dSPM' # dSPM or MNE or sLORETA
depth = None
mode_labels = 'mean_flip'

folder_fl = op.join(target, 'results', './parcellations')
string_target_file = op.join(folder_fl,
                        '{:s}_flame_grad_k{:d}_gamma{:1.2f}_theta{:1.2f}.pkl')

# In[] Initialization
eucl_errors_full = []
eucl_errors = {x : [] for x in \
    ['k%d_gamma%1.2f'%(knn, gamma) for knn in knn_tot for gamma in gamma_tot]}
eucl_errors_full_subs = {sub: eucl_errors_full for sub in subjects} 
eucl_errors_subs = {sub: {x : [] for x in \
    ['k%d_gamma%1.2f'%(knn, gamma) for knn in knn_tot for gamma in gamma_tot]}
                    for sub in subjects} 

correct_roi_an = np.zeros(len(subjects))
correct_roi_an_full = np.zeros(len(subjects))
correct_roi = np.zeros((np.size(knn_tot), np.size(gamma_tot), len(subjects)))
correct_roi_full = np.zeros((np.size(knn_tot), np.size(gamma_tot), len(subjects)))

dist_index_an = np.zeros(len(subjects))
dist_index = np.zeros((np.size(knn_tot), np.size(gamma_tot), len(subjects)))

# +
for idx_sub, subject in enumerate(subjects):
    
    print('Working with %s'%subject)
    
# In[]: Step 2. Load 
#   2.a. Forward model   
    path_lf = string_lf.format(subject)
    fwd = read_forward_solution(path_lf)
    fwd = pick_types_forward(fwd, meg=sensors_meg, 
                eeg=False, ref_meg=False, exclude='bads')
    fwd = convert_forward_solution(fwd, 
                surf_ori=True, force_fixed=True, use_cps=True)

    #   2.b. Anatomical regions
    label_lh = read_labels_from_annot(subject=subject, parc=parc, hemi='lh',
                           subjects_dir=subjects_dir)
    label_rh = read_labels_from_annot(subject=subject, parc=parc, hemi='rh', 
                        subjects_dir=subjects_dir)
    label = label_lh + label_rh
    
    #   2.c. Parameters for inverse modeling
    n_vert = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']
    L = fwd['sol']['data']
    sigma = np.mean(np.mean(L**2, axis=0)) / snr
    C = sigma * np.eye(L.shape[0])

# In[]: Step 3. Analysis with anatomical regions.
    proj_anrois = megicparc.labels_to_array(label, fwd['src'])
    if proj_anrois['outliers'] > 0:
        aux_num_roi = len(proj_anrois['parcel']) - 1
    else:
        aux_num_roi = len(proj_anrois['parcel'])

#       3.1. Source estimation and localization error on full source space 
    W_full = compute_inv_op_rank(L, C, lam, 
                        depth=depth, method=method, rank=None)
    R_full = np.dot(W_full, L)
    peak_full = np.argmax(R_full**2, axis=0)

    ee_full = megicparc.compute_localization_error(peak_full, 
                                     np.arange(n_vert), fwd['src'])
    eucl_errors_full += ee_full.tolist()
    
    eucl_errors_full_subs[subject] = ee_full

#       3.2. RM on anatomical regions and related evaluation criteria
    R_an = collapse_RM(R_full, label, fwd['src'], mode_labels)

    peak_c_an = np.argmax(abs(R_an), axis=0)
    true_c_an = megicparc.membership2vector( 
                        proj_anrois['parcel'][:aux_num_roi], n_vert)
    correct_roi_an[idx_sub] = np.count_nonzero(true_c_an == peak_c_an)/n_vert*100
    correct_roi_an_full[idx_sub] = np.count_nonzero(true_c_an[peak_full] == peak_c_an) \
                                    /n_vert*100                                   
    _, dist_index_an[idx_sub] = megicparc.compute_distinguishability_index(
                        R_an, proj_anrois['parcel'], aux_num_roi)

# In[]: Step 4. Analysis with meg-informed parcels
    for idx_g, gamma in enumerate(gamma_tot):
        for idx_k, knn in enumerate(knn_tot):
            
            target_file = string_target_file.format(
                            subject, knn, gamma, theta)
            print('Loading %s'%target_file)
            with open(target_file, 'rb') as aux_lf:
                flame_data = pickle.load(aux_lf)

            # 4.1. Compute RMs
            L_fl = L[:, flame_data['centroids_id']]
            W_fl = compute_inv_op_rank(L_fl, C, lam, 
                                     depth=depth, method=method)
            R_fl = np.dot(W_fl, L)

            peak_c = np.argmax(abs(R_fl), axis=0)

            # 4.2. Compute localization errors
            ee = megicparc.compute_localization_error(peak_c, 
                                    flame_data['centroids_id'], fwd['src'])
            eucl_errors['k%d_gamma%1.2f'%(knn, gamma)] += ee.tolist()

            eucl_errors_subs[subject]['k%d_gamma%1.2f'%(knn, gamma)] = ee

            # 4.3. Correctly identified regions
            true_c = megicparc.membership2vector( 
                     flame_data['parcel'][:flame_data['centroids']], n_vert)
            correct_roi[idx_k, idx_g, idx_sub] = \
                np.count_nonzero(true_c == peak_c)/n_vert*100
            correct_roi_full[idx_k, idx_g, idx_sub] = \
                np.count_nonzero(true_c[peak_full] == peak_c)/n_vert*100

            # 4.4. compute DI
            _, DI = megicparc.compute_distinguishability_index(
                    R_fl, flame_data['parcel'], flame_data['centroids'])
            dist_index[idx_k, idx_g, idx_sub] = DI
            
            del flame_data


# %%
# In[]: Step 5. Averages
Ns = len(subjects)

# 5.1. Anatomic parcels
aux_ee = np.asanyarray(eucl_errors_full)
eucl_err_full_ave = np.mean(aux_ee)
eucl_err_full_sem = np.std(aux_ee, ddof=1) / np.sqrt(np.size(aux_ee))

correct_roi_an_ave = np.mean(correct_roi_an)
correct_roi_an_sem = np.std(correct_roi_an, ddof=1) / np.sqrt(Ns)

correct_roi_an_full_ave = np.mean(correct_roi_an_full)
correct_roi_an_full_sem = np.std(correct_roi_an_full, ddof=1) / np.sqrt(Ns)

dist_index_an_ave = np.mean(dist_index_an)
dist_index_an_sem = np.std(dist_index_an, ddof=1) / np.sqrt(Ns)

# 5.2. MEG-informed parcels    
correct_roi_ave = np.mean(correct_roi, axis=2)
correct_roi_sem = np.std(correct_roi, axis=2, ddof=1) / np.sqrt(Ns)

correct_roi_full_ave = np.mean(correct_roi_full, axis=2)
correct_roi_full_sem = np.std(correct_roi_full, axis=2, ddof=1) / np.sqrt(Ns)

dist_index_ave = np.mean(dist_index, axis=2)
dist_index_sem = np.std(dist_index, axis=2, ddof=1) / np.sqrt(Ns)
    
eucl_err_ave = np.zeros((np.size(knn_tot), np.size(gamma_tot)))
eucl_err_sem = np.zeros((np.size(knn_tot), np.size(gamma_tot)))

for idx_g, gamma in enumerate(gamma_tot):
    for idx_k, knn in enumerate(knn_tot):

        aux_ee = np.asanyarray(eucl_errors['k%d_gamma%1.2f'%(knn, gamma)])
        eucl_err_ave[idx_k, idx_g] = np.mean(aux_ee)
        eucl_err_sem[idx_k, idx_g] = np.std(aux_ee, ddof=1) / np.sqrt(np.size(aux_ee))

# %%
# In[]: Step 6. Statistical test

# 6.1. Distinguishability index
di_t_scores = np.zeros((np.size(knn_tot), np.size(gamma_tot)))
di_p_values = np.zeros((np.size(knn_tot), np.size(gamma_tot)))

for idx_k, knn in enumerate(knn_tot):
    for idx_g, gamma in enumerate(gamma_tot):

        [aux_t, aux_p]= stats.ttest_rel(dist_index[idx_k, idx_g, :],
                                        dist_index_an, alternative='greater')
        di_t_scores[idx_k, idx_g] = aux_t
        di_p_values[idx_k, idx_g] = aux_p


# %%
# In[]: Step 7. Plot
plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{color}')
plt.rc('font', family='serif', size=22)
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['errorbar.capsize'] = 3.5
plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams['patch.force_edgecolor'] = True

colors = np.array([np.array([0, 1, 0]), 
                  np.array([0, 0, 1]), 
                  np.array([1, 0, 0]), 
                  np.array([0, 0.9655, 1])])
color_grey = np.array([112, 128, 144])/256

# %%
# 7.1. Euclidean localization error
f_ee = plt.figure()
#plt.errorbar(gamma_tot, eucl_err_full_ave*np.ones(np.size(gamma_tot)), 
#             yerr=eucl_err_full_sem*np.ones(np.size(gamma_tot)),
#             fmt='k--', label='full source-space')
plt.errorbar(gamma_tot, eucl_err_full_ave*np.ones(np.size(gamma_tot)), 
            yerr=eucl_err_full_sem*np.ones(np.size(gamma_tot)),
            color=color_grey, linestyle='dashed', label='full source-space')
for idx_k, knn in enumerate(knn_tot):
    plt.errorbar(gamma_tot, 
                 eucl_err_ave[idx_k, :], yerr=eucl_err_sem[idx_k, :],
                 label='$k$ = %d'%knn, color=colors[idx_k])
plt.xlabel(r'Weight of the spatial distances', fontsize=30)
plt.ylabel(r'Localization error (mm)', fontsize=30)
plt.xlim(-0.1, 1.1)
if method == 'dSPM': 
    plt.ylim(10, 25)
else:
    plt.ylim(0, 27)
plt.xticks(gamma_tot)
f_ee.set_size_inches(6.5, 7)
plt.tight_layout(pad=1)
f_ee.savefig(op.join(path_fig, 'eucl_loc_err_%s.png'%method))

# 7.2. Percentage of corrected identified regions
f_corr = plt.figure()
for idx_k, knn in enumerate(knn_tot):
    plt.errorbar(gamma_tot, 
       correct_roi_ave[idx_k, :], yerr=correct_roi_sem[idx_k, :], 
       label='$k$ = %d'%knn, color=colors[idx_k])
plt.errorbar(gamma_tot, 
   correct_roi_an_ave*np.ones(np.size(gamma_tot)),
   yerr=correct_roi_an_sem*np.ones(np.size(gamma_tot)),
   fmt='k--', label='DK')
plt.xlabel(r'Weight of the spatial distances', fontsize=30)
plt.xticks(gamma_tot)
plt.ylabel(r'Correctly estimated locations ($\%$)',fontsize=30)
plt.ylim(10, 70)
plt.xlim(-0.1, 1.1)
f_corr.set_size_inches(6.5, 7)
plt.tight_layout(pad=1)
f_corr.savefig(op.join(path_fig, 'corrected_regions_%s.png'%method))


f_corr_full = plt.figure()
ax_corr_full = f_corr_full.add_subplot(1,1,1)
for idx_k, knn in enumerate(knn_tot):
    ax_corr_full.errorbar(gamma_tot, 
       correct_roi_full_ave[idx_k, :], yerr=correct_roi_full_sem[idx_k, :], 
       label='$k$ = %d'%knn, color=colors[idx_k])
ax_corr_full.errorbar(gamma_tot, 
   correct_roi_an_full_ave*np.ones(np.size(gamma_tot)),
   yerr=correct_roi_an_full_sem*np.ones(np.size(gamma_tot)),
   color=color_grey, linestyle='dashed', label='Full source space')
ax_corr_full.errorbar(gamma_tot, 
   correct_roi_an_full_ave*np.ones(np.size(gamma_tot)),
   yerr=correct_roi_an_full_sem*np.ones(np.size(gamma_tot)),
   fmt='k--', label='DK')
ax_corr_full.set_xlabel(r'Weight of the spatial distances', fontsize=30)
ax_corr_full.set_xticks(gamma_tot)
ax_corr_full.set_ylabel(r'Correctly estimated locations  ($\%$)',fontsize=30)
ax_corr_full.set_ylim(10, 70)
ax_corr_full.set_xlim(-0.1, 1.1)
f_corr_full.set_size_inches(6.5, 7)
plt.tight_layout(pad=1)
f_corr_full.savefig(op.join(path_fig, 'corrected_regions_%s_full.png'%method))

label_params = ax_corr_full.get_legend_handles_labels()
figl, axl = plt.subplots()
axl.axis(False)
axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size":12}, ncol=6)
figl.set_size_inches(10, 0.5)
plt.tight_layout(pad=1.5)

figl.savefig(op.join(path_fig, 'loc_err_legend.png'))

# 7.3. Distinhuishability index
_aux = np.array([-0.01, 0.005, -0.005, 0.01])
f_DI = plt.figure()
for idx_k, knn in enumerate(knn_tot):
    plt.errorbar(gamma_tot+_aux[idx_k], dist_index_ave[idx_k, :], 
                 yerr=dist_index_sem[idx_k, :], 
                 label='$k$ = %d'%knn, color=colors[idx_k], 
                 zorder=idx_k)
plt.errorbar(gamma_tot, 
         dist_index_an_ave*np.ones(np.size(gamma_tot)), 
         yerr=dist_index_an_sem*np.ones(np.size(gamma_tot)), fmt='k--', 
         label='DK')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',fontsize=18)
plt.xlabel(r'Weight of the spatial distances', fontsize=25)
plt.ylabel(r'Distinguishability index', fontsize=25)
plt.ylim(0, 1)
plt.xlim(-0.05, 1.05)
plt.xticks(gamma_tot)

f_DI.set_size_inches(9, 5.5)
plt.tight_layout(pad=1.5)
f_DI.savefig(op.join(path_fig, 'dist_index_%s.png'%method))



# %%
# Additional plots
# AP1. Statistical test
f_test = plt.figure()
for idx_k, knn in enumerate(knn_tot):
    plt.plot(gamma_tot, di_p_values[idx_k, :], 
                 label='$k$ = %d'%knn, color=colors[idx_k], 
                 zorder=idx_k)
plt.yscale('log')
plt.legend()
plt.xlabel(r'Weight of the spatial distances', fontsize=30)
plt.ylabel(r'p value', fontsize=30)
plt.xlim(-0.05, 1.05)

f_test.set_size_inches(9, 5.5)
plt.tight_layout(pad=1.5)
f_test.savefig(op.join(path_fig, 'dist_index_%s_ttest.png'%method))


# %%
print(di_p_values<0.001)
print(di_p_values)

# %%
# AP2. Spatial distribution of the localization error
#clim = {'kind' : 'value', 'lims' : [0, 0.3, 1]}
subjects = ['k1_T1', 'k2_T1', 'k4_T1', 'k6_T1', 'k7_T1',
           'CC110045', 'CC110182', 'CC110174', 'CC110126', 'CC110056']

clmap = 'hot'
for idx_sub, subj_sd in enumerate(subjects):
    knn_sd = [20, 30]
    gamma_sd = [0.20, 0.40]
    
    path_lf = string_lf.format(subj_sd)
    fwd = read_forward_solution(path_lf)
    vertices_plot = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    
    for idx_knn, knn in enumerate(knn_sd):
        for idx_gamma, gamma in enumerate(gamma_sd):
            
            target_file = string_target_file.format(subj_sd, knn, gamma, theta)
            print('Loading %s'%target_file)
            with open(target_file, 'rb') as aux_lf:
                flame_data = pickle.load(aux_lf)
            
            aux_ee = eucl_errors_subs[subj_sd]['k%d_gamma%1.2f'%(knn, gamma)]
            aux_ee = aux_ee / np.max(aux_ee)
            clim = {'kind' : 'value', 'lims' : [0, 0.3, 1]}
            #clim = {'kind' : 'value', 'lims' : [0, np.mean(aux_ee), 1*np.max(aux_ee)]}
            stc_ee = SourceEstimate(aux_ee[:, np.newaxis], vertices_plot, 
                                      tmin = 0, tstep = 1, subject=subj_sd)
            brain = stc_ee.plot(views="lat", hemi="split", size=(800, 400),
                            subject=subj_sd, subjects_dir=subjects_dir, background="w", 
                            colorbar=False, clim=clim, time_viewer=False, show_traces=False, 
                            colormap=clmap)
            plot_flame_centroids(flame_data, fwd['src'], subj_sd, 
                        subjects_dir, brain=brain, scale_factor=0.65)
            screenshot = brain.screenshot()
            brain.close()
            nonwhite_pix = (screenshot != 255).any(-1)
            nonwhite_row = nonwhite_pix.any(1)
            nonwhite_col = nonwhite_pix.any(0)
            cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(cropped_screenshot)
            ax.axis("off")
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.2)
            #cbar = plot_brain_colorbar(cax, clim, label='Norm. Loc. Err.')
            fig.subplots_adjust(left=0.05, right=0.9, bottom=0.01, top=0.9, 
                                wspace=0.1, hspace=0.5)
            fig.savefig(op.join(path_fig,  
                'sp_locerr_%s_k%d_gamma%1.2f_fl.png'%(subj_sd, knn, gamma)))
            
            del aux_ee, stc_ee, brain, clim
    
    aux_ee_full = eucl_errors_full_subs[subj_sd]
    aux_ee_full = aux_ee_full / np.max(aux_ee_full)
    clim = {'kind' : 'value', 'lims' : [0, 0.3, 1]}
    #clim = {'kind' : 'value', 'lims' : [0, np.mean(aux_ee_full), 1*np.max(aux_ee_full)]}
    stc_ee_full = SourceEstimate(aux_ee_full[:, np.newaxis], vertices_plot, 
                            tmin = 0, tstep = 1, subject=subj_sd)
    brain = stc_ee_full.plot(views="lat", hemi="split", size=(800, 400),
                    subject=subj_sd, subjects_dir=subjects_dir, background="w", 
                    colorbar=False, clim=clim, time_viewer=False, show_traces=False,
                    colormap=clmap)
    screenshot = brain.screenshot()
    brain.close()
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    
    fig, ax = plt.subplots(figsize=(8.5, 3))
    ax.imshow(cropped_screenshot)
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plot_brain_colorbar(cax, clim, label='Norm. Loc. Err.', colormap='autumn')
    fig.subplots_adjust(left=0.05, right=0.85, bottom=0.01, top=0.9, 
                        wspace=0.1, hspace=0.5)
    fig.savefig(op.join(path_fig,  
        'sp_locerr_%s_full.png'%(subj_sd)))
