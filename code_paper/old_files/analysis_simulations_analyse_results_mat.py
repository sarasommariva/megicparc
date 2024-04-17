# Author: Sara Sommariva <sara.sommariva@aalto.fi>

# Objective: Run simulation for flame paper

# Objective 1. Perform only the analysis of the results, by using already 
#              computed estimates
# Objective 2. Merge with the code of data generation and analysis and perform 
#              again the whole pipeline
# Objective 3. Perform more complicated simulations? 

# TODO: controllo sul numero di run utili?
# TODO: controllo sull'errore relativo non in valore assoluto
# TODO: plottare non solo le medie, ma anche gli istogrammi dei vali criteri di valutazione
# TODO: fare parte con la coerenza

import os.path as op
import numpy as np
import scipy.io

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../code/')
import f_inverse

# In[]: Step 1. Define general parameters
target = '/m/nbe/work/sommars1/FLAME/'
subjects = ['k1_T1', 'k2_T1', 'k4_T1', 'k6_T1', 'k7_T1', 
           'CC110045', 'CC110182', 'CC110174', 'CC110126', 'CC110056']

folder_results = op.join(target, 'code_simulation_analysis/mypipeline_data_an')
string_results = op.join(folder_results, '{:s}', 
                         'result_grad_alpha_bio_{:s}_num_{:d}_{:s}')
path_true_val = op.join(folder_results, 'x_mvar.mat')

path_fig = './figures'

gamma_tot = np.arange(0, 1.01, 0.2)
knn_tot = [10, 20, 30, 40]
alpha_bio_tot = [0.25, 0.5]
num_run = 100

# In[]: Initialization
true_ic = np.zeros((num_run, np.size(subjects)))

est_ic_an = np.full((num_run, np.size(subjects), np.size(alpha_bio_tot)), np.nan)
rel_err_ic_an = np.full((num_run, np.size(subjects), np.size(alpha_bio_tot)), np.nan)
ratio_strongest_an = np.full((num_run, np.size(subjects), np.size(alpha_bio_tot)), np.nan)
tpr_an = np.zeros((num_run, np.size(subjects), np.size(alpha_bio_tot)))
fpr_an = np.full((num_run, np.size(subjects), np.size(alpha_bio_tot)), np.nan)

est_ic = np.full((np.size(knn_tot), np.size(gamma_tot), num_run,
                   np.size(subjects), np.size(alpha_bio_tot)), np.nan)
rel_err_ic = np.full((np.size(knn_tot), np.size(gamma_tot), num_run,
                   np.size(subjects), np.size(alpha_bio_tot)), np.nan)
ratio_strongest = np.full((np.size(knn_tot), np.size(gamma_tot), num_run,
                   np.size(subjects), np.size(alpha_bio_tot)), np.nan)
tpr = np.zeros((np.size(knn_tot), np.size(gamma_tot), num_run,
                   np.size(subjects), np.size(alpha_bio_tot)))
fpr = np.full((np.size(knn_tot), np.size(gamma_tot), num_run,
                   np.size(subjects), np.size(alpha_bio_tot)), np.nan)

auc_ic_an = np.full((num_run, np.size(subjects), np.size(alpha_bio_tot)), np.nan)
auc_coh_an = np.full((num_run, np.size(subjects), np.size(alpha_bio_tot)), np.nan)
auc_ic = np.full((np.size(knn_tot), np.size(gamma_tot), num_run,
                   np.size(subjects), np.size(alpha_bio_tot)), np.nan)
auc_coh = np.full((np.size(knn_tot), np.size(gamma_tot), num_run,
                   np.size(subjects), np.size(alpha_bio_tot)), np.nan)

# In[]: Step 2. Load
for idx_sub, subject in enumerate(subjects):
    for i_run in range(num_run):
        
#   2.a. True values (put here for the sake of generalization, but the true values is always the same)
        _aux = scipy.io.loadmat(path_true_val)
        true_ic[i_run, idx_sub] = _aux['mean_ic'][1, 0]
        
        for idx_a, alpha_bio in enumerate(alpha_bio_tot):

#   2.b. Estimated connectivity values
            path_res = string_results.format(subject, 
                                str(alpha_bio), i_run+1, subject)
            print('Loading %s'%path_res)
            _aux_est = scipy.io.loadmat(path_res)
            wc_tot = np.arange(11, 0, -2)
            
            idx_roi_an = _aux_est['dipoles']['anroi_idx'][0, 0][0] - 1
            idx_roi = {'k%d_g%1.1f'%(knn, gamma_tot[ig]) : \
                _aux_est['dipoles']['centroid_idx'][0, 0][
                        'knn_%d_wc_%d'%(knn, wc_tot[ig])][0, 0][0] - 1 \
                for knn in knn_tot for ig in range(np.size(gamma_tot))}
            
            est_ic_an_mat = _aux_est['conn_est']['ic_an_mean'][0, 0]
            est_coh_an_mat = _aux_est['conn_est']['coh_an_mean'][0, 0]
            est_ic_mat = {'k%d_g%1.1f'%(knn, gamma_tot[ig]) : \
                _aux_est['conn_est']['ic_fl_mean'][0, 0][
                        'knn_%d_wc_%d'%(knn, wc_tot[ig])][0, 0] \
                for knn in knn_tot for ig in range(np.size(gamma_tot))}
            est_coh_mat = {'k%d_g%1.1f'%(knn, gamma_tot[ig]) : \
                _aux_est['conn_est']['coh_fl_mean'][0, 0][
                        'knn_%d_wc_%d'%(knn, wc_tot[ig])][0, 0] \
                for knn in knn_tot for ig in range(np.size(gamma_tot))}
            
            sign_values_an = _aux_est['conn_est']['sign_values_mc_an'][0, 0]
            sign_values = {'k%d_g%1.1f'%(knn, gamma_tot[ig]) : \
                _aux_est['conn_est']['sign_values_mc'][0, 0][
                        'knn_%d_wc_%d'%(knn, wc_tot[ig])][0, 0] \
                for knn in knn_tot for ig in range(np.size(gamma_tot))}
            
            del _aux_est
            
# In[]: Step 3. Analysis with anatomical Atlas
            idx_roi_an = np.sort(idx_roi_an)[::-1]
            n_roi_an = est_ic_an_mat.shape[0]
            
            if min(idx_roi_an) > -1 and np.size(np.unique(idx_roi_an)) == 2: 
                # --> exclude run in which the two interecting sources are outliers
                #     or belong to the same region.
                
#   3.a. Relative error in estimating the connection of interest
                _aux_ic = est_ic_an_mat[idx_roi_an[0], idx_roi_an[1]]
                est_ic_an[i_run, idx_sub, idx_a] = _aux_ic
                rel_err_ic_an[i_run, idx_sub, idx_a] = \
                    abs(_aux_ic - true_ic[i_run, idx_sub]) / abs(true_ic[i_run, idx_sub])
#   3.b. Strongest connection
                ratio_strongest_an[i_run, idx_sub, idx_a] = \
                    _aux_ic / est_ic_an_mat.max()
                    
#   3.c. Sensitivity and specificity
                tpr_an[i_run, idx_sub, idx_a] = sign_values_an[
                                            idx_roi_an[0], idx_roi_an[1]]
                fpr_an[i_run, idx_sub, idx_a] = \
                    (sign_values_an.sum() - tpr_an[i_run, idx_sub, idx_a]) / \
                    (0.5 * n_roi_an * (n_roi_an - 1) - 1)
                    
#   3.d. Roc analysis for both ic and coh
                auc_ic_an[i_run, idx_sub, idx_a] = \
                    f_inverse.compute_auc(est_ic_an_mat, idx_roi_an)
                    
                auc_coh_an[i_run, idx_sub, idx_a] = \
                    f_inverse.compute_auc(est_coh_an_mat, idx_roi_an)
                
            else: # FPR are always computed!
                
                fpr_an[i_run, idx_sub, idx_a] = \
                    sign_values_an.sum() / (0.5 * n_roi_an * (n_roi_an - 1))
                
# In[]: Step 4. Analysis with flame parcellations
            for idx_g, gamma in enumerate(gamma_tot):
                for idx_k, knn in enumerate(knn_tot):
                    
                    _idx_roi = np.sort(idx_roi['k%d_g%1.1f'%(knn, gamma)])[::-1]
                    n_roi = est_ic_mat['k%d_g%1.1f'%(knn, gamma)].shape[0] 

#   4.a. Relative error in estimating the connection of interest
                    if min(_idx_roi) > -1 and np.size(np.unique(_idx_roi)) == 2:
                # --> exclude run in which the two interecting sources are outliers
                #     or belong to the same region.
                        _aux_ic = est_ic_mat['k%d_g%1.1f'%(knn, gamma)][
                            _idx_roi[0], _idx_roi[1]]
                        est_ic[idx_k, idx_g, i_run, idx_sub, idx_a] = _aux_ic
                        rel_err_ic[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            abs(_aux_ic - true_ic[i_run, idx_sub]) / abs(true_ic[i_run, idx_sub])
#   4.b. Strongest connection
                        ratio_strongest[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            _aux_ic / est_ic_mat['k%d_g%1.1f'%(knn, gamma)].max()
#   4.c. Sensitivity and specificity
                        _aux_sign = sign_values['k%d_g%1.1f'%(knn, gamma)]
                        tpr[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            _aux_sign[_idx_roi[0], _idx_roi[1]]
                        fpr[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            (_aux_sign.sum()-tpr[idx_k, idx_g, i_run, idx_sub, idx_a]) / \
                            (0.5 * n_roi * (n_roi-1) - 1)
                            
#   4.d. Roc analysis for both ic and coh
                        auc_ic[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            f_inverse.compute_auc(est_ic_mat['k%d_g%1.1f'%(knn, gamma)], _idx_roi)
                        auc_coh[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            f_inverse.compute_auc(est_coh_mat['k%d_g%1.1f'%(knn, gamma)], _idx_roi)
                            
                    else:
                        
                        fpr[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            _aux_sign.sum() / (0.5 * n_roi * (n_roi-1) - 1)
                            
# In[] Step 5. Averages
                            
#    5.a Run with interacting sources in different regions
run_correct_an = np.invert(np.isnan(est_ic_an[:, :, 0]))
nc_run_an = np.sum(run_correct_an, axis=0)
run_correct = np.invert(np.isnan(est_ic[:, :, :, :, 0]))
nc_run = np.sum(run_correct, axis=2)

#   5.b. Initialization
xi1_an_mean = np.zeros(np.size(alpha_bio_tot))
xi1_an_sem = np.zeros(np.size(alpha_bio_tot))
xi2_an_mean = np.zeros(np.size(alpha_bio_tot))
xi2_an_sem = np.zeros(np.size(alpha_bio_tot))
fpr_an_mean = np.zeros(np.size(alpha_bio_tot))
fpr_an_sem = np.zeros(np.size(alpha_bio_tot))

xi1_mean = np.zeros((np.size(knn_tot), np.size(gamma_tot), np.size(alpha_bio_tot)))
xi1_sem = np.zeros((np.size(knn_tot), np.size(gamma_tot), np.size(alpha_bio_tot)))
xi2_mean = np.zeros((np.size(knn_tot), np.size(gamma_tot), np.size(alpha_bio_tot)))
xi2_sem = np.zeros((np.size(knn_tot), np.size(gamma_tot), np.size(alpha_bio_tot)))
fpr_mean = np.zeros((np.size(knn_tot), np.size(gamma_tot), np.size(alpha_bio_tot)))
fpr_sem = np.zeros((np.size(knn_tot), np.size(gamma_tot), np.size(alpha_bio_tot)))

auc_ic_an_mean = np.zeros(np.size(alpha_bio_tot))
auc_ic_an_sem = np.zeros(np.size(alpha_bio_tot)) 
auc_coh_an_mean =  np.zeros(np.size(alpha_bio_tot))
auc_coh_an_sem =  np.zeros(np.size(alpha_bio_tot))

auc_ic_mean = np.zeros((np.size(knn_tot), np.size(gamma_tot), np.size(alpha_bio_tot)))
auc_ic_sem = np.zeros((np.size(knn_tot), np.size(gamma_tot), np.size(alpha_bio_tot)))
auc_coh_mean = np.zeros((np.size(knn_tot), np.size(gamma_tot), np.size(alpha_bio_tot)))
auc_coh_sem = np.zeros((np.size(knn_tot), np.size(gamma_tot), np.size(alpha_bio_tot))) 

for idx_a in range(np.size(alpha_bio_tot)):

#  5.c. Anatomical regions

    _var = rel_err_ic_an[run_correct_an, idx_a]
    xi1_an_mean[idx_a] = np.mean(_var)
    xi1_an_sem[idx_a] = np.std(_var) / np.sqrt(np.size(_var))
    del _var
    
    _var = ratio_strongest_an[run_correct_an, idx_a]
    xi2_an_mean[idx_a] = np.mean(_var)
    xi2_an_sem[idx_a] = np.std(_var) / np.sqrt(np.size(_var))
    del _var
    
    _var = fpr_an[run_correct_an, idx_a]
    fpr_an_mean[idx_a] = np.mean(_var)
    fpr_an_sem[idx_a] = np.std(_var) / np.sqrt(np.size(_var))
    del _var
    
    _var = auc_ic_an[run_correct_an, idx_a]
    auc_ic_an_mean[idx_a] = np.mean(_var)
    auc_ic_an_sem[idx_a] = np.std(_var) / np.sqrt(np.size(_var))
    del _var
    
    _var = auc_coh_an[run_correct_an, idx_a]
    auc_coh_an_mean[idx_a] = np.mean(_var)
    auc_coh_an_sem[idx_a] = np.std(_var) / np.sqrt(np.size(_var))
    del _var
    

#  5.d. Flame parcellations
    for idx_k in range(np.size(knn_tot)):
        for idx_g in range(np.size(gamma_tot)):
            
            _var = rel_err_ic[idx_k, idx_g, run_correct[idx_k, idx_g], idx_a]
            xi1_mean[idx_k, idx_g, idx_a] = np.mean(_var)
            xi1_sem[idx_k, idx_g, idx_a] = np.std(_var) / np.sqrt(np.size(_var))
            del _var
            
            _var = ratio_strongest[idx_k, idx_g, run_correct[idx_k, idx_g], idx_a]
            xi2_mean[idx_k, idx_g, idx_a] = np.mean(_var)
            xi2_sem[idx_k, idx_g, idx_a] = np.std(_var) / np.sqrt(np.size(_var))
            del _var
            
            _var = fpr[idx_k, idx_g, run_correct[idx_k, idx_g], idx_a]
            fpr_mean[idx_k, idx_g, idx_a] = np.mean(_var)
            fpr_sem[idx_k, idx_g, idx_a] = np.std(_var) / np.sqrt(np.size(_var))
            del _var
            
            _var = auc_ic[idx_k, idx_g, run_correct[idx_k, idx_g], idx_a]
            auc_ic_mean[idx_k, idx_g, idx_a] = np.mean(_var)
            auc_ic_sem[idx_k, idx_g, idx_a] = np.std(_var) / np.sqrt(np.size(_var))
            del _var
            
            _var = auc_coh[idx_k, idx_g, run_correct[idx_k, idx_g], idx_a]
            auc_coh_mean[idx_k, idx_g, idx_a] = np.mean(_var)
            auc_coh_sem[idx_k, idx_g, idx_a] = np.std(_var) / np.sqrt(np.size(_var))
            del _var


#   5.e. Specificity (averaged only over subjects)
tpr_an_perc = np.array([np.sum(tpr_an[:, :, i_a], axis=0)/nc_run_an*100 \
                        for i_a in range(np.size(alpha_bio_tot))])
tpr_an_perc_mean = np.mean(tpr_an_perc, axis=1)
tpr_an_perc_sem = np.std(tpr_an_perc, axis=1) / np.sqrt(np.size(subjects))


tpr_perc = np.array([np.array([np.array([
        np.sum(tpr[ik, ig, :, :, ia], axis=0) / nc_run[ik, ig] * 100 \
        for ia in range(np.size(alpha_bio_tot))])\
        for ig in range(np.size(gamma_tot))]) \
        for ik in range(np.size(knn_tot))]) 

tpr_perc_mean = np.mean(tpr_perc, axis=-1)
tpr_perc_sem = np.std(tpr_perc, axis=-1) / np.sqrt(np.size(subjects))

                        
# In[] Plots.
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['errorbar.capsize'] = 3.5
plt.rcParams["figure.figsize"] = [12, 8]

# In[]: Plot 1. Relative error in estimating the connection of interest
f_xi1, ax_xi1 = plt.subplots(1, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_xi1[idx_a].errorbar(gamma_tot+(idx_k-1)*0.005, 
               xi1_mean[idx_k, :, idx_a], yerr=xi1_sem[idx_k, :, idx_a], 
               label='k = %d'%knn)
    ax_xi1[idx_a].errorbar(gamma_tot+0.01, 
           xi1_an_mean[idx_a]*np.ones(np.size(gamma_tot)), 
           yerr=xi1_an_sem[idx_a]*np.ones(np.size(gamma_tot)), 
           fmt='k--', label='DK')
    
    ax_xi1[idx_a].set_ylim(0.15, 0.5)
    ax_xi1[idx_a].set_xlabel(r'$\gamma$')
    ax_xi1[idx_a].set_title(r'$\beta^b = %1.2f$'%alpha)
    ax_xi1[idx_a].set_xlim(-0.1, 1.1)


ax_xi1[0].set_ylabel(r'$\xi^{(1)}$')
lgd = ax_xi1[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_xi1.set_size_inches(20, 8)        
f_xi1.savefig(op.join(path_fig, 'rel_err_xi1.png'), 
              bbox_extra_artists=(lgd,), bbox_inches='tight')

# In[]: Plot 2. Ratio over the strongest connection
f_xi2, ax_xi2 = plt.subplots(1, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_xi2[idx_a].errorbar(gamma_tot+(idx_k-1)*0.005, 
               xi2_mean[idx_k, :, idx_a], yerr=xi2_sem[idx_k, :, idx_a], 
               label='k = %d'%knn)
    ax_xi2[idx_a].errorbar(gamma_tot+0.01, 
           xi2_an_mean[idx_a]*np.ones(np.size(gamma_tot)), 
           yerr=xi2_an_sem[idx_a]*np.ones(np.size(gamma_tot)), 
           fmt='k--', label='DK')
    
    ax_xi2[idx_a].set_ylim(0.5, 0.8)
    ax_xi2[idx_a].set_xlabel(r'$\gamma$')
    ax_xi2[idx_a].set_title(r'$\beta^b = %1.2f$'%alpha)
    ax_xi2[idx_a].set_xlim(-0.1, 1.1)

ax_xi2[0].set_ylabel(r'$\xi^{(2)}$')
lgd = ax_xi2[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_xi2.set_size_inches(20, 8)        
f_xi2.savefig(op.join(path_fig, 'ratio_strongest_conn_xi2.png'), 
              bbox_extra_artists=(lgd,), bbox_inches='tight')


# In[]: Plot 3. False positive rate 
f_fpr, ax_fpr = plt.subplots(1, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_fpr[idx_a].errorbar(gamma_tot+(idx_k-1)*0.005, 
               fpr_mean[idx_k, :, idx_a], yerr=fpr_sem[idx_k, :, idx_a], 
               label='k = %d'%knn)
    ax_fpr[idx_a].errorbar(gamma_tot+0.01, 
           fpr_an_mean[idx_a]*np.ones(np.size(gamma_tot)), 
           yerr=fpr_an_sem[idx_a]*np.ones(np.size(gamma_tot)), 
           fmt='k--', label='DK')
        
    ax_fpr[idx_a].set_xlabel(r'$\gamma$')
    ax_fpr[idx_a].set_title(r'$\beta^b = %1.2f$'%alpha)
    ax_fpr[idx_a].set_xlim(-0.1, 1.1)

ax_fpr[0].set_ylabel(r'FPR')
ax_fpr[0].set_ylim(0, 0.15)
ax_fpr[1].set_ylim(0, 0.06)

lgd = ax_fpr[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_fpr.set_size_inches(20, 8)        
f_fpr.savefig(op.join(path_fig, 'fpr.png'), 
              bbox_extra_artists=(lgd,), bbox_inches='tight')


# In[]: Plot 4. Percentage of dataset where we recognise the connection of interest
f_tpr, ax_tpr = plt.subplots(1, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_tpr[idx_a].errorbar(gamma_tot+(idx_k-1)*0.005, 
               tpr_perc_mean[idx_k, :, idx_a], 
               yerr=tpr_perc_sem[idx_k, :, idx_a], label='k = %d'%knn)
    ax_tpr[idx_a].errorbar(gamma_tot+0.01, 
           tpr_an_perc_mean[idx_a]*np.ones(np.size(gamma_tot)), 
           yerr=tpr_an_perc_sem[idx_a]*np.ones(np.size(gamma_tot)), 
           fmt='k--', label='DK')
        
    ax_tpr[idx_a].set_xlabel(r'$\gamma$')
    ax_tpr[idx_a].set_title(r'$\beta^b = %1.2f$'%alpha)
    ax_tpr[idx_a].set_ylim(30, 100)
    ax_tpr[idx_a].set_xlim(-0.1, 1.1)

ax_tpr[0].set_ylabel(r'TPR')
ax_tpr[1].legend()

lgd = ax_tpr[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_tpr.set_size_inches(20, 8)        
f_tpr.savefig(op.join(path_fig, 'perc_tpr.png'), 
              bbox_extra_artists=(lgd,), bbox_inches='tight')

# In[]: Plot 5. 
f_auc, ax_auc = plt.subplots(2, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_auc[idx_a, 0].errorbar(gamma_tot+(idx_k-1)*0.005, 
               auc_ic_mean[idx_k, :, idx_a], 
               yerr=auc_ic_sem[idx_k, :, idx_a], label='k = %d'%knn)
        ax_auc[idx_a, 1].errorbar(gamma_tot+(idx_k-1)*0.005, 
               auc_coh_mean[idx_k, :, idx_a], 
               yerr=auc_coh_sem[idx_k, :, idx_a], label='k = %d'%knn)
    
    ax_auc[idx_a, 0].errorbar(gamma_tot+0.01, 
           auc_ic_an_mean[idx_a]*np.ones(np.size(gamma_tot)), 
           yerr=auc_ic_an_sem[idx_a]*np.ones(np.size(gamma_tot)), 
           fmt='k--', label='DK')    
    ax_auc[idx_a, 1].errorbar(gamma_tot+0.01, 
           auc_coh_an_mean[idx_a]*np.ones(np.size(gamma_tot)), 
           yerr=auc_coh_an_sem[idx_a]*np.ones(np.size(gamma_tot)), 
           fmt='k--', label='DK')    
    
    ax_auc[idx_a, 0].set_ylim(0.6, 1)
    ax_auc[idx_a, 1].set_ylim(0.6, 1)
    ax_auc[idx_a, 0].set_ylabel(r'AUC $\beta^b = %1.2f$'%alpha)
       
for ii in range(ax_auc.shape[1]):
    ax_auc[-1, ii].set_xlabel(r'$\gamma$')
ax_auc[0, 0].set_title('IC')
ax_auc[0, 1].set_title('COH')

    
lgd = ax_auc[0, -1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_auc.set_size_inches(20, 16) 
f_auc.savefig(op.join(path_fig, 'auc_ic_coh.png'), 
              bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()
#    ax_tpr[idx_a].errorbar(gamma_tot+0.01, 
#           tpr_an_perc_mean[idx_a]*np.ones(np.size(gamma_tot)), 
#           yerr=tpr_an_perc_sem[idx_a]*np.ones(np.size(gamma_tot)), 
#           fmt='k--', label='DK')



                    
            
