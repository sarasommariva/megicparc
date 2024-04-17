# Author: Sara Sommariva <sara.sommariva@aalto.fi>

# Objective: Compute flame parcellations and/or analyse their structural 
# and functional properties via resolution matrices

from __future__ import division
import os.path as op
import os
import numpy as np
from scipy import stats

import pickle

from mne import (read_forward_solution, pick_types_forward, 
                 convert_forward_solution, read_labels_from_annot)
from mne.viz import get_brain_class

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
sys.path.insert(0, '../code/')
import flame
import f_inverse
        
# Options:
# (1) Do you want to compute and save new parcellations? 
#     If 'True' the code only compute and save the required parcellations
#     If 'False' precomputed parcellations will be used and the analysis will 
#     be performed
do_compute_flame = False
# (2) Do you want to plot also figures that require pysurfer? 
do_brain_plot = True
# (3) Do you want to save figure? 
do_save_fig = True
path_fig = './figures'

# In[]: Step 1. Define general parameters
target = '/m/nbe/work/sommars1/FLAME/'
subjects = ['k1_T1', 'k2_T1', 'k4_T1', 'k6_T1', 'k7_T1', 
           'CC110045', 'CC110182', 'CC110174', 'CC110126', 'CC110056']
subjects_dir = op.join(target, 'subjects_flame')
sensors_meg = 'grad' 
string_lf = op.join('./data', 
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

folder_fl = op.join('./flame_parcellations')
string_target_file = op.join(folder_fl,
                        '{:s}_flame_grad_k{:d}_gamma{:1.2f}_theta{:1.2f}.pkl')

# Creating some folders (if not existing)
if not op.isdir(path_fig):
    os.mkdir(path_fig)
if not op.isdir(folder_fl):
    os.mkdir(folder_fl)

# In[] Initialization
if not do_compute_flame:
    num_roi_an = np.zeros(len(subjects))
    num_roi = np.zeros((np.size(knn_tot), np.size(gamma_tot), len(subjects)))
    
    n_cc = {x : [] for x in \
            ['k%d_gamma%1.2f'%(knn, gamma) for knn in knn_tot for gamma in gamma_tot]}
    
    coverage_an = np.zeros(len(subjects))
    coverage = np.zeros((np.size(knn_tot), np.size(gamma_tot), len(subjects)))
    
    eucl_errors_full = []
    eucl_errors = {x : [] for x in \
        ['k%d_gamma%1.2f'%(knn, gamma) for knn in knn_tot for gamma in gamma_tot]}

    correct_roi_an = np.zeros(len(subjects))
    correct_roi_an_full = np.zeros(len(subjects))
    correct_roi = np.zeros((np.size(knn_tot), np.size(gamma_tot), len(subjects)))
    correct_roi_full = np.zeros((np.size(knn_tot), np.size(gamma_tot), len(subjects)))
    
    dist_index_an = np.zeros(len(subjects))
    dist_index = np.zeros((np.size(knn_tot), np.size(gamma_tot), len(subjects)))


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

    # In[]: Step 3. Run flame 
    if do_compute_flame: 
    
        for gamma in gamma_tot:
            sort_dist = flame.compute_distance_matrix(fwd, gamma=gamma, 
                                              theta=theta, labels=label)
            for knn in knn_tot:
                target_file = string_target_file.format(
                                subject, knn, gamma, theta)
                flame_data = flame.compute_parcellation(sort_dist, k_nn=knn)
                # - Save         
                print('Saving %s'%target_file)
                aux_f = open(target_file, 'wb')
                pickle.dump(flame_data, aux_f, protocol=2)
                aux_f.close()
            
                del flame_data
            del sort_dist
            
        if subject == 'k2_T1': # For 'k2_T1' we also compute parcellations for theta = 0
            
            theta_aux = 0
            
            for gamma in gamma_tot:
                sort_dist = flame.compute_distance_matrix(fwd, gamma=gamma, 
                                              theta=theta_aux, labels=label)
                for knn in knn_tot:
                    target_file = string_target_file.format(
                                subject, knn, gamma, theta_aux)
                    flame_data = flame.compute_parcellation(sort_dist, k_nn=knn)
                    # - Save         
                    print('Saving %s'%target_file)
                    aux_f = open(target_file, 'wb')
                    pickle.dump(flame_data, aux_f, protocol=2)
                    aux_f.close()
            
                    del flame_data
                del sort_dist
             
    else:
        
    #   or perform analysis from precomputed parcellation
        n_vert = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']
        L = fwd['sol']['data']
        sigma = np.mean(np.mean(L**2, axis=0)) / snr
        C = sigma * np.eye(L.shape[0])
        
    # In[]: Step 4. Analysis with anatomical regions.
        proj_anrois = flame.labels_to_array(label, fwd['src'])
        
    # In[]: 4.a. DK - structural properties
    #       4.a.1. Number of regions
        if proj_anrois['outliers'] > 0:
            aux_num_roi = len(proj_anrois['parcel']) - 1
        else:
            aux_num_roi = len(proj_anrois['parcel'])
        num_roi_an[idx_sub] = aux_num_roi
    #       4.a.2. Coverage
        coverage_an[idx_sub] = 100 * (n_vert - proj_anrois['outliers']) / n_vert
#        
    # In[]: 4.b. DK - functional properties
    #       4.b.1. Source estimation and localization error on full source space 
        W_full = f_inverse.compute_inv_op_rank(L, C, lam, 
                            depth=depth, method=method, rank=None)
        R_full = np.dot(W_full, L)
        peak_full = np.argmax(R_full**2, axis=0)
        
        ee_full = flame.compute_localization_error(peak_full, 
                                         np.arange(n_vert), fwd['src'])
        eucl_errors_full += ee_full.tolist()
        
    #       4.b.2. RM on anatomical regions and related evaluation criteria
        R_an = f_inverse.collapse_RM(R_full, label, fwd['src'], mode_labels)
        
        peak_c_an = np.argmax(abs(R_an), axis=0)
        true_c_an = flame.membership2vector( 
                            proj_anrois['parcel'][:aux_num_roi], n_vert)
        correct_roi_an[idx_sub] = np.count_nonzero(true_c_an == peak_c_an)/n_vert*100
        correct_roi_an_full[idx_sub] = np.count_nonzero(true_c_an[peak_full] == peak_c_an) \
                                        /n_vert*100                                   
        _, dist_index_an[idx_sub] = flame.compute_distinguishability_index(
                            R_an, proj_anrois['parcel'], aux_num_roi)
#        
#    # In[]: Step 5. Initialization

        bmesh_adj_mat = flame.triangulation2adjacency(fwd['src'])
#        
       
        for idx_g, gamma in enumerate(gamma_tot):
            for idx_k, knn in enumerate(knn_tot):
                
                target_file = string_target_file.format(
                                subject, knn, gamma, theta)
                print('Loading %s'%target_file)
                with open(target_file, 'rb') as aux_lf:
                    flame_data = pickle.load(aux_lf)
                    
    # In[]: Step 6. Structural properties:
                    # 6.a. Number of regions
                    num_roi[idx_k, idx_g, idx_sub] = flame_data['centroids']
                    # 6.b. Outliers
                    coverage[idx_k, idx_g, idx_sub] = \
                            100 * (n_vert - flame_data['outliers']) / n_vert
                    # 6.c. Number of connected components
                    n_cc['k%d_gamma%1.2f'%(knn, gamma)] += \
                         [len(flame.compute_connected_components(
                                 bmesh_adj_mat, flame_data['parcel'][ir])) \
                                 for ir in range(flame_data['centroids'])]
    
    # In[]: Step 7. Functional properties (RM analysis)     
                    # 7.a. Compute RMs
                    L_fl = L[:, flame_data['centroids_id']]
                    W_fl = f_inverse.compute_inv_op_rank(L_fl, C, lam, 
                                             depth=depth, method=method)
                    R_fl = np.dot(W_fl, L)
                    
                    peak_c = np.argmax(abs(R_fl), axis=0)
                    
                    # 7.b. Compute localization errors
                    ee = flame.compute_localization_error(peak_c, 
                                            flame_data['centroids_id'], fwd['src'])
                    eucl_errors['k%d_gamma%1.2f'%(knn, gamma)] += ee.tolist()
                           
                    
                    # 7.c. Correctly identified regions
                    true_c = flame.membership2vector( 
                             flame_data['parcel'][:flame_data['centroids']], n_vert)
                    correct_roi[idx_k, idx_g, idx_sub] = \
                        np.count_nonzero(true_c == peak_c)/n_vert*100
                    correct_roi_full[idx_k, idx_g, idx_sub] = \
                        np.count_nonzero(true_c[peak_full] == peak_c)/n_vert*100
                    
                    # 7.d. compute DI
                    _, DI = flame.compute_distinguishability_index(
                            R_fl, flame_data['parcel'], flame_data['centroids'])
                    dist_index[idx_k, idx_g, idx_sub] = DI
                    
                    del flame_data
                    
# In[]: Step 8. Averages
if not do_compute_flame:
    
    Ns = len(subjects)
    
    # 8.a. Anatomic regions
    num_roi_an_ave = np.mean(num_roi_an)
    num_roi_an_sem = np.std(num_roi_an, ddof=1) / np.sqrt(Ns)
    
    aux_ee = np.asanyarray(eucl_errors_full)
    eucl_err_full_ave = np.mean(aux_ee)
    eucl_err_full_sem = np.std(aux_ee, ddof=1) / np.sqrt(np.size(aux_ee))

    correct_roi_an_ave = np.mean(correct_roi_an)
    correct_roi_an_sem = np.std(correct_roi_an, ddof=1) / np.sqrt(Ns)

    correct_roi_an_full_ave = np.mean(correct_roi_an_full)
    correct_roi_an_full_sem = np.std(correct_roi_an_full, ddof=1) / np.sqrt(Ns)
    
    dist_index_an_ave = np.mean(dist_index_an)
    dist_index_an_sem = np.std(dist_index_an, ddof=1) / np.sqrt(Ns)

    # 8.b. Flame
    num_roi_ave = np.mean(num_roi, axis=2)    
    num_roi_sem = np.std(num_roi, axis=2, ddof=1) / np.sqrt(Ns) 
    
    correct_roi_ave = np.mean(correct_roi, axis=2)
    correct_roi_sem = np.std(correct_roi, axis=2, ddof=1) / np.sqrt(Ns)
    
    correct_roi_full_ave = np.mean(correct_roi_full, axis=2)
    correct_roi_full_sem = np.std(correct_roi_full, axis=2, ddof=1) / np.sqrt(Ns)
    
    dist_index_ave = np.mean(dist_index, axis=2)
    dist_index_sem = np.std(dist_index, axis=2, ddof=1) / np.sqrt(Ns)
    
    
    eucl_err_ave = np.zeros((np.size(knn_tot), np.size(gamma_tot)))
    eucl_err_sem = np.zeros((np.size(knn_tot), np.size(gamma_tot)))
    n_cc_ave = np.zeros((np.size(knn_tot), np.size(gamma_tot)))
    n_cc_sem = np.zeros((np.size(knn_tot), np.size(gamma_tot)))
    
    for idx_g, gamma in enumerate(gamma_tot):
        for idx_k, knn in enumerate(knn_tot):

            aux_ee = np.asanyarray(eucl_errors['k%d_gamma%1.2f'%(knn, gamma)])
            eucl_err_ave[idx_k, idx_g] = np.mean(aux_ee)
            eucl_err_sem[idx_k, idx_g] = np.std(aux_ee, ddof=1) / np.sqrt(np.size(aux_ee))
            aux_cc = np.asanyarray(n_cc['k%d_gamma%1.2f'%(knn, gamma)])
            n_cc_ave[idx_k, idx_g] = np.mean(aux_cc)
            n_cc_sem[idx_k, idx_g] = np.std(aux_cc, ddof=1) / np.sqrt(np.size(aux_cc))
            
# In[]: Step 9. Statistical test

#       9.a. Distinguishability index
    di_t_scores = np.zeros((np.size(knn_tot), np.size(gamma_tot)))
    di_p_values = np.zeros((np.size(knn_tot), np.size(gamma_tot)))

    for idx_k, knn in enumerate(knn_tot):
        for idx_g, gamma in enumerate(gamma_tot):

            [aux_t, aux_p]= stats.ttest_rel(dist_index[idx_k, idx_g, :],
                                            dist_index_an, alternative='greater')
            di_t_scores[idx_k, idx_g] = aux_t
            di_p_values[idx_k, idx_g] = aux_p

# In[]: Step 10. Impact of the anatomical constraints
    ac_sub = 'k2_T1'
    
    path_lf = string_lf.format(ac_sub)
    fwd = read_forward_solution(path_lf)
    
    label_lh = read_labels_from_annot(subject=ac_sub, parc=parc, hemi='lh',
                           subjects_dir=subjects_dir)
    label_rh = read_labels_from_annot(subject=ac_sub, parc=parc, hemi='rh', 
                    subjects_dir=subjects_dir)
    label = label_lh + label_rh
    proj_anrois = flame.labels_to_array(label, fwd['src'])
    
    del label, fwd
    
    k_nn_ac = [30]
    theta_ac = [0, 0.05]
    gamma_ac = np.arange(1, 0.4, -0.2)
    
    tolls = [0.05]
    
    spread = {}

    num_roi_constr = {}

    for idx_t, theta in enumerate(theta_ac):
        for idx_k, knn in enumerate(k_nn_ac):
            for idx_g, gamma in enumerate(gamma_ac):
#           Step 1. Reload parcellation                 
                target_file = string_target_file.format(ac_sub, knn, gamma, 
                                                        theta)
                print('Loading %s'%target_file)
                with open(target_file, 'rb') as aux_lf:
                    flame_data = pickle.load(aux_lf)

                num_roi_constr['k%d_gamma%1.2f_theta%1.2f'%(knn, gamma, theta)] = \
                    flame_data['centroids']
#           Step 2. Compute cotingency matrix
                conting_mat = np.array([
                        np.array([
                        np.intersect1d(p_fl, p_an).shape[0] 
                        for p_an in proj_anrois['parcel']]) 
                        for p_fl in flame_data['parcel'][0:flame_data['centroids']]])
                conting_mat = conting_mat / \
                        np.sum(conting_mat, axis=1)[:, np.newaxis]
#           Step 3. Compute spread of flame regions over anatomical
                spread['k%d_gamma%1.2f_theta%1.2f'%(knn, gamma, theta)] = \
                    np.zeros((flame_data['centroids'], np.size(tolls)))
                    
                for idx_t, toll in enumerate(tolls):
                    spread['k%d_gamma%1.2f_theta%1.2f'%(knn, gamma, theta)] [:, idx_t] = \
                        np.sum(conting_mat > toll, axis=1)
   
# In[] Plots
    
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
    
    # In[]. Plot 1. Number of regions
    f1 = plt.figure()
    for idx_k, knn in enumerate(knn_tot):
        plt.errorbar(gamma_tot, num_roi_ave[idx_k, :],
                     yerr=num_roi_sem[idx_k, :], color=colors[idx_k], 
                     label='$k$ = %d'%knn)
    plt.errorbar(gamma_tot, num_roi_an_ave*np.ones(np.size(gamma_tot)),
                 yerr=num_roi_an_sem*np.ones(np.size(gamma_tot)), 
                 fmt='k--', label='DK')
    plt.legend(loc=[0.78, 0.53])
    plt.xlabel(r'Weight of the spatial distances', fontsize=30)
    plt.ylabel(r'Number of parcels', fontsize=30)
    plt.ylim(0, 350)
    plt.xlim(-0.05, 1.05)
    
    if do_save_fig:
        f1.savefig(op.join(path_fig, 'num_regions.png'))
    
    # In[]. Plot 2. (Additional) Coverage
    f2 = plt.figure()
    for idx_k, knn in enumerate(knn_tot):
        aux_xax =np.concatenate([gamma*np.ones(Ns) for gamma in gamma_tot])
        plt.plot(aux_xax, coverage[idx_k].reshape(Ns*np.size(gamma_tot)), 
                     '.', color=colors[idx_k], label='k = %d'%knn)
    #plt.plot(gamma_tot+0.01, coverage_an*np.ones(np.size(gamma_tot)), 'k.', 
    #            label='DK', linewidth=2)
    plt.legend()
    plt.xlabel(r'Weight of the spatial distances', fontsize=30)
    plt.ylabel(r'Coverage', fontsize=30)
    plt.xlim(-0.1, 1.1)

    # In[]. Plot 3. Number of connected components
    f3 = plt.figure()
    for idx_k, knn in enumerate(knn_tot):
        plt.errorbar(gamma_tot, n_cc_ave[idx_k, :], yerr=n_cc_sem[idx_k, :],
                     label='$k$ = %d'%knn, color=colors[idx_k])
    plt.legend()
    plt.xlabel(r'Weight of the spatial distances', fontsize=30)
    plt.ylabel(r'Number connected components', fontsize=30)
    plt.ylim(0, 30)
    plt.xlim(-0.05, 1.05)
    #plt.show()
    
    if do_save_fig:
        f3.savefig(op.join(path_fig, 'n_connected_comp.png'))
    
    
    # In[]. Plot 5. Impact anatomical constraints
    bins = np.arange(-0.5, 69, 1)
    alpha = 0.7
    
    knn = k_nn_ac[0]
    toll = tolls[0]
        
    fac, axac = plt.subplots(1, np.size(gamma_ac), figsize=[13, 5])
    #fac.suptitle('$k =$ %d'%knn)
    idx_t = 0

    for idx_g, gamma in enumerate(gamma_ac):
        aux_th1 = spread['k%d_gamma%1.2f_theta%1.2f'%(knn, gamma, theta_ac[1])][:, idx_t] 
        axac[idx_g].hist(aux_th1, bins=bins, alpha=alpha,
                        weights=np.ones_like(aux_th1)/np.shape(aux_th1)[0]*100, 
                        label=r'$\theta = $ %1.2f'%theta_ac[1])
        aux_th2 = spread['k%d_gamma%1.2f_theta%1.2f'%(knn, gamma, theta_ac[0])] [:, idx_t]
        axac[idx_g].hist(aux_th2, bins=bins, alpha=alpha, 
                        weights=np.ones_like(aux_th2)/np.shape(aux_th2)[0]*100, 
                        label=r'$\theta = $ %1.2f'%theta_ac[0])
        axac[idx_g].set_ylim(0, 100)
        axac[idx_g].set_xlim(-0.5, 10.5)
        #axac[idx_t, idx_g].legend()
        
        if idx_t == 0:
            axac[idx_g].set_title(r'$\gamma = %1.1f$ '%(gamma)) 
            axac[idx_g].set_xticks(np.arange(1, 11, 2))
            axac[idx_g].text(6.5, 85, '%d ;'%(num_roi_constr['k%d_gamma%1.2f_theta0.05'%(knn, gamma)]), 
                             ha='center', color='dodgerblue', fontweight='bold')
            axac[idx_g].text(8.5, 85, '%d'%(num_roi_constr['k%d_gamma%1.2f_theta0.00'%(knn, gamma)]), 
                             ha='center', color='darkorange', fontweight='bold')
            axac[idx_g].text(7.2, 75, 'parcels', ha='center')
            rect = patches.FancyBboxPatch((5, 72), 4.5, 22, 
                                      boxstyle='round', facecolor='white', 
                                     edgecolor='lightgray', zorder=2)
            axac[idx_g].add_patch(rect)
        else:
            axac[idx_g].set_xticks(np.arange(1, 11, 2))
                    
        if idx_g == 0:
            axac[idx_g].set_ylabel(
                r'Number of parcels ($\%$)', 
                             fontsize=30)
        else:
            axac[idx_g].set_yticklabels([])
    
    #hand, lab = axac[idx_g].get_legend_handles_labels()
    #fac.legend(hand, lab, loc=[0.9, 0.5])
    axac[-1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
    fac.text(0.45, 0.02, 'Number of intersecting atlas parcels', ha='center', 
                  fontsize=30)
    plt.tight_layout(pad=1.5)
    
        
    if do_save_fig:            
        fac.savefig(op.join(path_fig, 'an_constr_%s_k%d'%(ac_sub, knn)))

       # plt.show()

    # In[]. Plot 5b. (Additional) impact of theta
    gamma_tplot = np.arange(0, 1, 0.1)
    theta_tplot = np.array([0.01, 0.025, 0.05, 0.1, 0.2])
    f_tg = np.array([theta * (gamma_tplot / (1-gamma_tplot)) \
                 for theta in theta_tplot])

    colors_tplot = np.array([np.array([0, 1, 0]),
            np.array([0, 0, 1]), np.array([1, 0, 0]),
            np.array([0, 0.9655, 1]), np.array([1, 0.4828, 0.8621])])

    f_tplot = plt.figure()
    for idx_t, theta in enumerate(theta_tplot):
        plt.plot(gamma_tplot, f_tg[idx_t],
             label=r'$\theta$ = %1.3f'%theta, color=colors_tplot[idx_t])
    plt.legend()
    plt.xlim([0, 0.9])
    plt.ylim([0, 2])
    plt.xlabel(r'$\gamma$', fontsize=30)
    plt.ylabel(r'$\theta$ $\frac{\gamma}{1-\gamma}$', fontsize=30)

    if do_save_fig:
       f_tplot.savefig(op.join(path_fig, 'impact_theta.png')) 

        
    # In[]. Plot 4. Euclidean localization errors
    f4 = plt.figure()
    plt.errorbar(gamma_tot, eucl_err_full_ave*np.ones(np.size(gamma_tot)), 
                 yerr=eucl_err_full_sem*np.ones(np.size(gamma_tot)),
                 fmt='k--', label='full source-space')
    for idx_k, knn in enumerate(knn_tot):
        plt.errorbar(gamma_tot, 
                     eucl_err_ave[idx_k, :], yerr=eucl_err_sem[idx_k, :],
                     label='$k$ = %d'%knn, color=colors[idx_k])
    plt.legend()
    plt.xlabel(r'Weight of the spatial distances', fontsize=30)
    plt.ylabel(r'Localization error (mm)', fontsize=30)
    plt.xlim(-0.1, 1.1)
    if method == 'dSPM': 
        plt.ylim(10, 30)
    else:
        plt.ylim(0, 27)
    
    if do_save_fig:
        f4.savefig(op.join(path_fig, 'eucl_loc_err_%s.png'%method))
    
    #plt.show()
    
    # In[]. Plot 5. Percentage of corrected identified regions
    f5, ax5 = plt.subplots(1, 2)
    for idx_k, knn in enumerate(knn_tot):
        ax5[0].errorbar(gamma_tot, 
           correct_roi_ave[idx_k, :], yerr=correct_roi_sem[idx_k, :], 
           label='$k$ = %d'%knn, color=colors[idx_k])
        ax5[1].errorbar(gamma_tot, 
           correct_roi_full_ave[idx_k, :], yerr=correct_roi_full_sem[idx_k, :], 
           label='$k$ = %d'%knn, color=colors[idx_k])
    ax5[0].errorbar(gamma_tot, 
       correct_roi_an_ave*np.ones(np.size(gamma_tot)),
       yerr=correct_roi_an_sem*np.ones(np.size(gamma_tot)),
       fmt='k--', label='DK')
    ax5[1].errorbar(gamma_tot, 
       correct_roi_an_full_ave*np.ones(np.size(gamma_tot)),
       yerr=correct_roi_an_full_sem*np.ones(np.size(gamma_tot)),
       fmt='k--', label='DK')
    ax5[1].legend()
    ax5[0].set_xlabel(r'Weight of the \\ spatial distances', fontsize=30)
    ax5[1].set_xlabel(r'Weight of the \\ spatial distances', fontsize=30)
    ax5[0].set_xticks(gamma_tot)
    ax5[1].set_xticks(gamma_tot)
    ax5[0].set_ylabel(r'Number of source locations ($\%$)',fontsize=30)
    
    pmin = 10
    pmax = 100
    ax5[0].set_ylim(10, 100)
    ax5[1].set_ylim(10, 100)
    ax5[0].set_xlim(-0.1, 1.1)
    ax5[1].set_xlim(-0.1, 1.1)
    plt.tight_layout()
    
    if do_save_fig:
        f5.savefig(op.join(path_fig, 'corrected_regions_%s.png'%method))
    
    # In[]. Plot 6. Distinhuishability index
    _aux = np.array([-0.01, 0.005, -0.005, 0.01])
    f6 = plt.figure()
    for idx_k, knn in enumerate(knn_tot):
        plt.errorbar(gamma_tot+_aux[idx_k], dist_index_ave[idx_k, :], 
                     yerr=dist_index_sem[idx_k, :], 
                     label='$k$ = %d'%knn, color=colors[idx_k], 
                     zorder=idx_k)
    plt.errorbar(gamma_tot, 
             dist_index_an_ave*np.ones(np.size(gamma_tot)), 
             yerr=dist_index_an_sem*np.ones(np.size(gamma_tot)), fmt='k--', 
             label='DK')
    plt.legend()
    plt.xlabel(r'Weight of the spatial distances', fontsize=30)
    plt.ylabel(r'Distinguishability index', fontsize=30)
    plt.ylim(0, 1)
    plt.xlim(-0.05, 1.05)
    
    if do_save_fig:
        f6.savefig(op.join(path_fig, 'dist_index_%s.png'%method))
    
    plt.show()

# In[]. Plot 6b. Statistical test
    f6_tt = plt.figure()
    for idx_k, knn in enumerate(knn_tot):
        plt.plot(gamma_tot, di_p_values[idx_k, :], 
                     label='$k$ = %d'%knn, color=colors[idx_k], 
                     zorder=idx_k)
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r'Weight of the spatial distances', fontsize=30)
    plt.ylabel(r'p value', fontsize=30)
    plt.xlim(-0.05, 1.05)
    
    if do_save_fig:
        f6_tt.savefig(op.join(path_fig, 'dist_index_%s_ttest.png'%method))


# In[]. Plots on brain
    if do_brain_plot:
        
        Brain = get_brain_class()
        
        ex_sub = 'k2_T1'
        
        path_lf = string_lf.format(ex_sub)
        fwd = read_forward_solution(path_lf)
        
        label_lh = read_labels_from_annot(subject=ex_sub, parc=parc, hemi='lh',
                           subjects_dir=subjects_dir)
        label_rh = read_labels_from_annot(subject=ex_sub, parc=parc, hemi='rh', 
                        subjects_dir=subjects_dir)
        label = label_lh + label_rh
            
# In[] Plot brain 1. Spatial cohesion
        knn_plot = 30
        gamma_plot = [0, 0.4, 0.6, 1]
        theta_plot = 0.05
        cart_coord = [-0.03, 0.025, 0.08]
  
        src = fwd['src']
        V = np.concatenate((src[0]['rr'][src[0]['vertno']],
                       src[1]['rr'][src[1]['vertno']]), axis=0)
        nvert = src[0]['nuse'] + src[1]['nuse']
        
        idx_vert = np.argmin(np.linalg.norm(V - cart_coord, axis=1))
            
        for gamma in gamma_plot:
            target_file = string_target_file.format(
                                ex_sub, knn_plot, gamma, theta_plot)
            print('Loading %s'%target_file)
            with open(target_file, 'rb') as aux_lf:
                flame_data = pickle.load(aux_lf)
            flame_labels = flame.store_flame_labels(flame_data, src, ex_sub)
                
            # TODO: make a function? (given a point select to whic region it belongs)
            parcels_vector = np.zeros(nvert, dtype='int') - 1
            for ir in range(flame_data['centroids']):
                parcels_vector[flame_data['parcel'][ir]] = ir
            sel_roi = parcels_vector[idx_vert]
            index = [sel_roi + 1]
            
# In[] Plot brain 1. Spatial cohesion
        knn_plot = 30
        gamma_plot = [0, 0.4, 0.6, 1]
        theta_plot = 0.05
        cart_coord = [-0.03, 0.025, 0.08]
  
        src = fwd['src']
        V = np.concatenate((src[0]['rr'][src[0]['vertno']],
                       src[1]['rr'][src[1]['vertno']]), axis=0)
        nvert = src[0]['nuse'] + src[1]['nuse']
        
        idx_vert = np.argmin(np.linalg.norm(V - cart_coord, axis=1))
            
        for gamma in gamma_plot:
            target_file = string_target_file.format(
                                ex_sub, knn_plot, gamma, theta_plot)
            print('Loading %s'%target_file)
            with open(target_file, 'rb') as aux_lf:
                flame_data = pickle.load(aux_lf)
            flame_labels = flame.store_flame_labels(flame_data, src, ex_sub)
                
            # TODO: make a function? (given a point select to whic region it belongs)
            parcels_vector = np.zeros(nvert, dtype='int') - 1
            for ir in range(flame_data['centroids']):
                parcels_vector[flame_data['parcel'][ir]] = ir
            sel_roi = parcels_vector[idx_vert]
            index = [sel_roi + 1]
            brain = flame.plot_flame_labels(index, flame_labels, src, ex_sub,
                    subjects_dir, surf='inflated', color = [1, 0.64, 0.],
                    plot_region=True, plot_points=False, plot_borders=False)
            brain.show_view({'azimuth': 173, 'elevation': 60})
            
            if do_save_fig:
                brain.save_image(op.join(path_fig, 
                    '%s_knn_%d_gamma_%1.2f.png'%(ex_sub, knn_plot, gamma)))

# In[]: Plot brain 2. Impact of the anatomical constraints
        knn_plot = 30
        gamma_plot = 0.8
        theta_plot = 0.05
        thresh = 0.05
        thresh_merg = 0.1

        target_file = string_target_file.format(
                                ex_sub, knn_plot, gamma_plot, theta_plot)
        print('Loading %s'%target_file)
        with open(target_file, 'rb') as aux_lf:
            flame_data = pickle.load(aux_lf)
            
        flame_labels = flame.store_flame_labels(flame_data, src, ex_sub)
            
        proj_anrois = flame.labels_to_array(label, src)
        
        conting_mat = np.array([
                        np.array([
                        np.intersect1d(p_fl, p_an).shape[0] 
                        for p_an in proj_anrois['parcel']]) 
                        for p_fl in flame_data['parcel'][0:flame_data['centroids']]])
        conting_mat = conting_mat / \
                        np.sum(conting_mat, axis=1)[:, np.newaxis]
                        
# In[].    2.1. Example of an anatomical regions splitted in two flame 
#                     regions            
        anat_sel = 'postcentral-lh'
        idx_an = proj_anrois['name'].index(anat_sel)
        if not label[idx_an].name == anat_sel:
            raise ValueError('Something wrong in selecting anatomical region.')
        
        idx_fl = np.where(conting_mat[:, idx_an]>thresh)[0]
        
        brain_split = Brain(ex_sub, hemi='both', surf='inflated', 
            background='white', subjects_dir=subjects_dir, alpha=1)
        brain_split.add_label(label[idx_an], hemi=label[idx_an].hemi, 
                              alpha=0.9) 
        aux = idx_fl+1
        flame.plot_flame_labels(aux.tolist(), flame_labels, src, ex_sub, 
            subjects_dir, surf='inflated', brain=brain_split, 
            plot_region=False, plot_points=True, plot_borders=False)
        
        nvert_lh = src[0]['nuse']
        idx_centr = flame_data['centroids_id'][idx_fl]
        centr_lh = idx_centr[np.where(idx_centr < nvert_lh)]
        centr_rh = idx_centr[np.where(idx_centr >= nvert_lh)] - nvert_lh
        
        if not centr_lh.size == 0:
            brain_split.add_foci(src[0]['vertno'][centr_lh], coords_as_verts=True, 
                    hemi='lh', scale_factor=0.7, color='white')
        if not centr_rh.size == 0:
            brain_split.add_foci(src[1]['vertno'][centr_rh], coords_as_verts=True, 
                    hemi='rh', scale_factor=0.7, color='white')
        
        brain_split.show_view({'azimuth': 180, 'elevation': 60})
        
        if do_save_fig:
            brain_split.save_image(op.join(path_fig, 
                    '%s_split_%d_%1.2f.png'%(ex_sub, knn_plot, gamma_plot)))

# In[].     2.2. Example of flame regions merged in anatomical rois
        anat_sel = 'parsopercularis-lh'
        idx_an = proj_anrois['name'].index(anat_sel)
        if not label[idx_an].name == anat_sel:
            raise ValueError('Something wrong in selecting anatomical region.')
        idx_fl = np.where(conting_mat[:, idx_an]>thresh)[0]
        idx_an_merged = np.array(
                [np.where(conting_mat[ir, :]>thresh_merg)[0] for ir in idx_fl])  
        idx_an_merged = np.unique(np.concatenate(idx_an_merged))
        
        idx_an_merged = idx_an_merged[1:]
        idx_fl = np.array([idx_fl[1]])
        
        brain_m = Brain(ex_sub, hemi='both', surf='inflated', 
                background='white', subjects_dir=subjects_dir, alpha=1)
        for ia in idx_an_merged:
            if ia < 68:
                brain_m.add_label(label[ia], hemi=label[ia].hemi, alpha=0.9) 
            else:
                print('@@@@@ Also outliers involved')

        colors = [0. , 1., 0.95600906, 1.] 
        for ii in range(idx_fl.shape[0]): # I need a for for selecting colors
            flame.plot_flame_labels([idx_fl[ii]+1], flame_labels, src, 
                ex_sub, subjects_dir, surf='inflated', brain=brain_m, 
                color=colors, plot_region=False, plot_points=True, 
                plot_borders=False)
            
        nvert_lh = src[0]['nuse']
        idx_centr = flame_data['centroids_id'][idx_fl]
        centr_lh = idx_centr[np.where(idx_centr < nvert_lh)]
        centr_rh = idx_centr[np.where(idx_centr >= nvert_lh)] - nvert_lh
        
        if not centr_lh.size == 0:
            brain_m.add_foci(src[0]['vertno'][centr_lh], coords_as_verts=True, 
                    hemi='lh', scale_factor=0.7, color='white')
        if not centr_rh.size == 0:
            brain_m.add_foci(src[1]['vertno'][centr_rh], coords_as_verts=True, 
                    hemi='rh', scale_factor=0.7, color='white')

        brain_m.show_view({'azimuth': 170, 'elevation': 90})
        
        if do_save_fig:
            brain_m.save_image(op.join(path_fig, 
                        '%s_merge_%d_%1.2f.png'%(ex_sub, knn_plot, gamma_plot)))

# In[].     2.3.  Reduced source space on top of DK atlas   
        brain_redV = Brain(ex_sub, hemi='both', surf='inflated', 
            background='white', subjects_dir=subjects_dir, alpha=1)
        #   Superimpose anatomical regions
        for ir in range(len(label)):
            brain_redV.add_label(label[ir], hemi=label[ir].hemi, alpha=0.9) 
    
        #  Superimpose centroids    
        flame.plot_flame_centroids(flame_data, src, ex_sub, subjects_dir, 
                                   brain_redV)

        brain_redV.show_view({'azimuth': 170, 'elevation': 90})
        
        if do_save_fig:
             brain_redV.save_image(op.join(path_fig, 
                    '%s_redV_%d_%1.2f.png'%(ex_sub, knn_plot, gamma_plot)))