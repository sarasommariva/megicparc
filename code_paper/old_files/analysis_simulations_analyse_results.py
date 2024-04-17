# Author: Sara Sommariva <sara.sommariva@aalto.fi>

# Objective: Run simulation for flame paper. Analysis of the results

# TODO: controllo sul numero di run utili?
# TODO: controllo sull'errore relativo non in valore assoluto
# TODO: plottare non solo le medie, ma anche gli istogrammi dei vali criteri di valutazione
# TODO: Check seed
# TODO: Cancella le variabili. Secondo me c'era un baco nel calcolo dei fpr
# TODO: Da remoto non posso utilizzate latex nelle immagini

import os.path as op
import os
import numpy as np
import pickle
from scipy import stats

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../code/')
import f_inverse # -> I need this for computing auc

# In[]: Step 1. Define general parameters
subjects = ['k1_T1', 'k2_T1', 'k4_T1', 'k6_T1', 'k7_T1', 
           'CC110045', 'CC110182', 'CC110174', 'CC110126', 'CC110056']

folder_results = op.join('./test_simulation')
string_results = op.join(folder_results, '{:s}',
                         'result_grad_num_{:d}_{:s}.pkl')
path_true_val = op.join(folder_results, 'x_mvar.pkl')

path_fig = './figures'
do_save_fig = True
if not op.isdir(path_fig):
    os.mkdir(path_fig)


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
        _aux = pickle.load(open(path_true_val, 'rb'))
        true_ic[i_run, idx_sub] = _aux['mean_ic'][1, 0]

# 2.b. Estimated connectivity values
        path_res = string_results.format(subject,
                                 i_run+1, subject)
        print('Loading %s' % path_res)
        _aux_est = pickle.load(open(path_res, 'rb'))

        idx_roi_an = _aux_est['dipoles']['an_regions']
        idx_roi = _aux_est['dipoles']['fl_regions']

        est_ic_an_mat = _aux_est['conn_est']['ic_an_mean']
        est_ic_mat = _aux_est['conn_est']['ic_fl_mean']

        sign_values_an = _aux_est['conn_est']['sign_values_an']
        sign_values = _aux_est['conn_est']['sign_values']

        est_coh_an_mat = _aux_est['conn_est']['coh_an_mean']
        est_coh_mat = _aux_est['conn_est']['coh_fl_mean']

        del _aux_est

        for idx_a, alpha_bio in enumerate(alpha_bio_tot):

# In[]: Step 3. Analysis with anatomical Atlas
            idx_roi_an = np.sort(idx_roi_an)[::-1]
            n_roi_an = est_ic_an_mat['ab_%1.2f'%alpha_bio].shape[0]
            
            if min(idx_roi_an) > -1 and np.size(np.unique(idx_roi_an)) == 2: 
                # --> exclude run in which the two interecting sources are outliers
                #     or belong to the same region.
                
#   3.a. Relative error in estimating the connection of interest
                _aux_ic = est_ic_an_mat['ab_%1.2f'%alpha_bio][
                                idx_roi_an[0], idx_roi_an[1]]
                est_ic_an[i_run, idx_sub, idx_a] = _aux_ic
                rel_err_ic_an[i_run, idx_sub, idx_a] = \
                    abs(_aux_ic - true_ic[i_run, idx_sub]) / abs(true_ic[i_run, idx_sub])
#   3.b. Strongest connection
                ratio_strongest_an[i_run, idx_sub, idx_a] = \
                    _aux_ic / est_ic_an_mat['ab_%1.2f'%alpha_bio].max()
                    
#   3.c. Sensitivity and specificity
                _aux_sign_an = sign_values_an['ab_%1.2f'%alpha_bio]
                tpr_an[i_run, idx_sub, idx_a] = _aux_sign_an[
                                            idx_roi_an[0], idx_roi_an[1]]
                fpr_an[i_run, idx_sub, idx_a] = \
                    (_aux_sign_an.sum() - tpr_an[i_run, idx_sub, idx_a]) / \
                    (0.5 * n_roi_an * (n_roi_an - 1) - 1)

#   3.d. Roc analysis for both ic and coh
                auc_ic_an[i_run, idx_sub, idx_a] = \
                    f_inverse.compute_auc(
                            est_ic_an_mat['ab_%1.2f'%alpha_bio], idx_roi_an)

                auc_coh_an[i_run, idx_sub, idx_a] = \
                    f_inverse.compute_auc(
                            est_coh_an_mat['ab_%1.2f'%alpha_bio], idx_roi_an)

            else: # FPR are always computed!

                _aux_sign_an = sign_values_an['ab_%1.2f' % alpha_bio]
                fpr_an[i_run, idx_sub, idx_a] = \
                    _aux_sign_an.sum() / (0.5 * n_roi_an * (n_roi_an - 1))
                
# In[]: Step 4. Analysis with flame parcellations
            for idx_g, gamma in enumerate(gamma_tot):
                for idx_k, knn in enumerate(knn_tot):
                    
                    _idx_roi = np.sort(idx_roi['k%d_gamma%1.2f'%(knn, gamma)])[::-1]
                    n_roi = est_ic_mat['k%d_gamma%1.2f_ab%1.2f'%(knn, gamma, alpha_bio)].shape[0] 

#   4.a. Relative error in estimating the connection of interest
                    if min(_idx_roi) > -1 and np.size(np.unique(_idx_roi)) == 2:
                # --> exclude run in which the two interecting sources are outliers
                #     or belong to the same region.
                        _aux_ic = est_ic_mat['k%d_gamma%1.2f_ab%1.2f'%(knn, gamma, alpha_bio)][
                            _idx_roi[0], _idx_roi[1]]
                        est_ic[idx_k, idx_g, i_run, idx_sub, idx_a] = _aux_ic
                        rel_err_ic[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            abs(_aux_ic - true_ic[i_run, idx_sub]) / abs(true_ic[i_run, idx_sub])
#   4.b. Strongest connection
                        ratio_strongest[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            _aux_ic / est_ic_mat['k%d_gamma%1.2f_ab%1.2f'%(knn, gamma, alpha_bio)].max()
#   4.c. Sensitivity and specificity
                        _aux_sign = sign_values['k%d_gamma%1.2f_ab%1.2f'%(knn, gamma, alpha_bio)]
                        tpr[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            _aux_sign[_idx_roi[0], _idx_roi[1]]
                        fpr[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            (_aux_sign.sum()-tpr[idx_k, idx_g, i_run, idx_sub, idx_a]) / \
                            (0.5 * n_roi * (n_roi-1) - 1)
#   4.d. Roc analysis for both ic and coh
                        auc_ic[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            f_inverse.compute_auc(
                                    est_ic_mat['k%d_gamma%1.2f_ab%1.2f'%(knn, gamma, alpha_bio)], _idx_roi)
                        auc_coh[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            f_inverse.compute_auc(
                                    est_coh_mat['k%d_gamma%1.2f_ab%1.2f'%(knn, gamma, alpha_bio)], _idx_roi)

                            
                    else:

                        _aux_sign = sign_values['k%d_gamma%1.2f_ab%1.2f' % (knn, gamma, alpha_bio)]
                        fpr[idx_k, idx_g, i_run, idx_sub, idx_a] = \
                            _aux_sign.sum() / (0.5 * n_roi * (n_roi-1))
                            
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
    
    _var = fpr_an[run_correct_an, idx_a]*100
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
            
            _var = fpr[idx_k, idx_g, run_correct[idx_k, idx_g], idx_a]*100
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

# In[] Step 6. Statistical test
xi1_p_values = np.zeros((np.size(knn_tot), np.size(gamma_tot),
                   np.size(subjects), np.size(alpha_bio_tot)))
xi2_p_values = np.zeros((np.size(knn_tot), np.size(gamma_tot),
                   np.size(subjects), np.size(alpha_bio_tot)))
fpr_p_values = np.zeros((np.size(knn_tot), np.size(gamma_tot),
                   np.size(subjects), np.size(alpha_bio_tot)))
tpr_p_values = np.zeros((np.size(knn_tot), np.size(gamma_tot),
                         np.size(alpha_bio_tot)))
n_run_ttest = np.zeros((np.size(knn_tot), np.size(gamma_tot),
                   np.size(subjects), np.size(alpha_bio_tot)))

for idx_a in range(np.size(alpha_bio_tot)):
    for idx_k in range(np.size(knn_tot)):
        for idx_g in range(np.size(gamma_tot)):
            # True Positive Rate
            [_, aux_p] = stats.ttest_rel(
                    tpr_perc[idx_k, idx_g, idx_a], 
                    tpr_an_perc[idx_a], alternative='greater')
            tpr_p_values[idx_k, idx_g, idx_a] = aux_p
            for idx_s in range(np.size(subjects)):
                aux_run = np.logical_and(run_correct[idx_k, idx_g, :, idx_s],
                                   run_correct_an[:, idx_s])
                n_run_ttest[idx_k, idx_g, idx_s, idx_a] = sum(aux_run)
            # Relative Error
                [_, aux_p] = stats.ttest_rel(
                    rel_err_ic[idx_k, idx_g, aux_run, idx_s, idx_a],
                    rel_err_ic_an[aux_run, idx_s, idx_a], alternative='less')
                xi1_p_values[idx_k, idx_g, idx_s, idx_a] = aux_p
            # Ratio with the strongest connection
                [_, aux_p] = stats.ttest_rel(
                    ratio_strongest[idx_k, idx_g, aux_run, idx_s, idx_a], 
                    ratio_strongest_an[aux_run, idx_s, idx_a],
                    alternative='greater')
                xi2_p_values[idx_k, idx_g, idx_s, idx_a] = aux_p
            # False positive rate
                [_, aux_p] = stats.ttest_rel(
                    fpr[idx_k, idx_g, aux_run, idx_s, idx_a],
                    fpr_an[aux_run, idx_s, idx_a], alternative='less')
                fpr_p_values[idx_k, idx_g, idx_s, idx_a] = aux_p

xi1_max_p_value = xi1_p_values.max(axis=2)
xi2_max_p_value = xi2_p_values.max(axis=2)
fpr_max_p_value = fpr_p_values.max(axis=2)

# In[] Plots.
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=30)
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['errorbar.capsize'] = 3.5
plt.rcParams["figure.figsize"] = [12, 8]

colors = np.array([np.array([0, 1, 0]), 
                   np.array([0, 0, 1]), 
                   np.array([1, 0, 0]), 
                   np.array([0, 0.9655, 1])])

# In[]: Plot 1. Relative error in estimating the connection of interest
f_xi1, ax_xi1 = plt.subplots(1, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_xi1[idx_a].errorbar(gamma_tot, 
               xi1_mean[idx_k, :, idx_a], yerr=xi1_sem[idx_k, :, idx_a], 
               label='$k$ = %d'%knn, color=colors[idx_k])
    ax_xi1[idx_a].errorbar(gamma_tot, 
           xi1_an_mean[idx_a]*np.ones(np.size(gamma_tot)), 
           yerr=xi1_an_sem[idx_a]*np.ones(np.size(gamma_tot)), 
           fmt='k--', label='DK')
    
    ax_xi1[idx_a].set_ylim(0.15, 0.5)
    ax_xi1[idx_a].set_xlim(-0.1, 1.1)
    ax_xi1[idx_a].set_xticks(gamma_tot)

ax_xi1[0].set_title(r'Low noise')
ax_xi1[1].set_title(r'High noise')
f_xi1.text(0.5, 0.015, 'Weight of the spatial distances', ha='center', 
                  fontsize=30)
ax_xi1[0].set_ylabel(r'Relative error', fontsize=30)
lgd = ax_xi1[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_xi1.set_size_inches(20, 8)
plt.tight_layout(pad=1.5)
if do_save_fig:
    f_xi1.savefig(op.join(path_fig, 'rel_err_xi1.png'))

# In[]: Plot 2. Ratio over the strongest connection
_aux = np.array([-0.01, 0.005, -0.005, 0.01])
f_xi2, ax_xi2 = plt.subplots(1, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_xi2[idx_a].errorbar(gamma_tot+_aux[idx_k], 
               xi2_mean[idx_k, :, idx_a], yerr=xi2_sem[idx_k, :, idx_a], 
               label='$k$ = %d'%knn, color=colors[idx_k])
    ax_xi2[idx_a].errorbar(gamma_tot, 
           xi2_an_mean[idx_a]*np.ones(np.size(gamma_tot)), 
           yerr=xi2_an_sem[idx_a]*np.ones(np.size(gamma_tot)), 
           fmt='k--', label='DK')
    ax_xi2[idx_a].set_xticks(gamma_tot)
    
    ax_xi2[idx_a].set_ylim(0.5, 0.8)
    ax_xi2[idx_a].set_xlim(-0.1, 1.1)

ax_xi2[0].set_title(r'Low noise')
ax_xi2[1].set_title(r'High noise')
f_xi2.text(0.5, 0.015, 'Weight of the spatial distances', ha='center', 
                  fontsize=30)
ax_xi2[0].set_ylabel(r'Ratio with the strongest connection', fontsize=30)
lgd = ax_xi2[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_xi2.set_size_inches(20, 8)
plt.tight_layout(pad=1.5)
if do_save_fig:  
    f_xi2.savefig(op.join(path_fig, 'ratio_strongest_conn_xi2.png'))


# In[]: Plot 3. False positive rate 
f_fpr, ax_fpr = plt.subplots(1, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_fpr[idx_a].errorbar(gamma_tot, 
               fpr_mean[idx_k, :, idx_a], yerr=fpr_sem[idx_k, :, idx_a], 
               label='$k$ = %d'%knn, color=colors[idx_k])
    ax_fpr[idx_a].errorbar(gamma_tot, 
           fpr_an_mean[idx_a]*np.ones(np.size(gamma_tot)), 
           yerr=fpr_an_sem[idx_a]*np.ones(np.size(gamma_tot)), 
           fmt='k--', label='DK')
    ax_fpr[idx_a].set_xlim(-0.1, 1.1)
    ax_fpr[idx_a].set_xticks(gamma_tot)
    
ax_fpr[0].set_title(r'Low noise')
ax_fpr[1].set_title(r'High noise')
ax_fpr[0].set_ylabel(r'False positive rate (\%)', fontsize=30)
ax_fpr[0].set_ylim(0, 14)
ax_fpr[1].set_ylim(0, 5)

lgd = ax_fpr[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_fpr.text(0.5, 0.015, 'Weight of the spatial distances', ha='center', 
                  fontsize=30)
f_fpr.set_size_inches(20, 8)
plt.tight_layout(pad=1.5)
if do_save_fig:      
    f_fpr.savefig(op.join(path_fig, 'fpr.png'))


# In[]: Plot 4. Percentage of dataset where we recognise the connection of interest
f_tpr, ax_tpr = plt.subplots(1, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_tpr[idx_a].errorbar(gamma_tot, 
               tpr_perc_mean[idx_k, :, idx_a], 
               yerr=tpr_perc_sem[idx_k, :, idx_a], label='$k$ = %d'%knn,
               color=colors[idx_k])
    ax_tpr[idx_a].errorbar(gamma_tot, 
           tpr_an_perc_mean[idx_a]*np.ones(np.size(gamma_tot)), 
           yerr=tpr_an_perc_sem[idx_a]*np.ones(np.size(gamma_tot)), 
           fmt='k--', label='DK')
    ax_tpr[idx_a].set_xticks(gamma_tot)
    ax_tpr[idx_a].set_ylim(30, 100)
    ax_tpr[idx_a].set_xlim(-0.1, 1.1)

ax_tpr[0].set_title(r'Low noise')
ax_tpr[1].set_title(r'High noise')
ax_tpr[0].set_ylabel(r'True positive rate (\%)', fontsize=30)
ax_tpr[1].legend()

lgd = ax_tpr[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_tpr.text(0.5, 0.015, 'Weight of the spatial distances', ha='center', 
                  fontsize=30)
f_tpr.set_size_inches(20, 8)
plt.tight_layout(pad=1.5)
if do_save_fig:      
    f_tpr.savefig(op.join(path_fig, 'perc_tpr.png'))
    
# In[]: Additional plots: Statistical test
#   A1. Relative Error
f_xi1_tt, ax_xi1_tt = plt.subplots(1, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_xi1_tt[idx_a].plot(gamma_tot, 
               xi1_max_p_value[idx_k, :, idx_a], 
               label='$k$ = %d'%knn, color=colors[idx_k])
        ax_xi1_tt[idx_a].set_yscale('log')
    ax_xi1_tt[idx_a].plot(gamma_tot, 
           0.05*np.ones(np.size(gamma_tot)), 
           'k--', label='p = 0.05')
    ax_xi1_tt[idx_a].set_xticks(gamma_tot)

ax_xi1_tt[0].set_title(r'Low noise')
ax_xi1_tt[1].set_title(r'High noise')
f_xi1_tt.text(0.5, 0.015, 'Weight of the spatial distances', ha='center', 
                  fontsize=30)
ax_xi1_tt[0].set_ylabel(r'p-value (Relative Error)', fontsize=30)

lgd = ax_xi1_tt[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_xi1_tt.set_size_inches(20, 8)
plt.tight_layout(pad=1.5)

if do_save_fig:
    f_xi1_tt.savefig(op.join(path_fig, 'rel_err_xi1_ttest.png'))

#   A2. Ratio with the strongest connection
f_xi2_tt, ax_xi2_tt = plt.subplots(1, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_xi2_tt[idx_a].plot(gamma_tot, 
               xi2_max_p_value[idx_k, :, idx_a], 
               label='$k$ = %d'%knn, color=colors[idx_k])
        ax_xi2_tt[idx_a].set_yscale('log')
    ax_xi2_tt[idx_a].plot(gamma_tot, 
           0.05*np.ones(np.size(gamma_tot)), 
           'k--', label='p = 0.05')
    ax_xi2_tt[idx_a].set_xticks(gamma_tot)

ax_xi2_tt[0].set_title(r'Low noise')
ax_xi2_tt[1].set_title(r'High noise')
f_xi2_tt.text(0.5, 0.015, 'Weight of the spatial distances', ha='center', 
                  fontsize=30)
ax_xi2_tt[0].set_ylabel(r'p-value (Ratio strongest)', fontsize=30)
lgd = ax_xi2_tt[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_xi2_tt.set_size_inches(20, 8)
plt.tight_layout(pad=1.5)

if do_save_fig:
    f_xi2_tt.savefig(op.join(path_fig, 'xi2_ttest.png'))

#   A3. False positive rate
f_fpr_tt, ax_fpr_tt = plt.subplots(1, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_fpr_tt[idx_a].plot(gamma_tot, 
               fpr_max_p_value[idx_k, :, idx_a], 
               label='$k$ = %d'%knn, color=colors[idx_k])
        ax_fpr_tt[idx_a].set_yscale('log')
    ax_fpr_tt[idx_a].plot(gamma_tot, 
           0.01*np.ones(np.size(gamma_tot)), 
           'k--', label='p = 0.01')
    ax_fpr_tt[idx_a].set_xticks(gamma_tot)

ax_fpr_tt[0].set_title(r'Low noise')
ax_fpr_tt[1].set_title(r'High noise')
f_fpr_tt.text(0.5, 0.015, 'Weight of the spatial distances', ha='center', 
                  fontsize=30)
ax_fpr_tt[0].set_ylabel(r'p-value (FPR)', fontsize=30)
lgd = ax_fpr_tt[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_fpr_tt.set_size_inches(20, 8)
plt.tight_layout(pad=1.5)

if do_save_fig:
    f_fpr_tt.savefig(op.join(path_fig, 'fpr_ttest.png'))

#   A4. True positive rate
f_tpr_tt, ax_tpr_tt = plt.subplots(1, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_tpr_tt[idx_a].plot(gamma_tot, 
               tpr_p_values[idx_k, :, idx_a], 
               label='$k$ = %d'%knn, color=colors[idx_k])
        ax_tpr_tt[idx_a].set_yscale('log')
    ax_tpr_tt[idx_a].plot(gamma_tot, 
           0.01*np.ones(np.size(gamma_tot)), 
           'k--', label='p = 0.01')
    ax_tpr_tt[idx_a].set_xticks(gamma_tot)

ax_tpr_tt[0].set_title(r'Low noise')
ax_tpr_tt[1].set_title(r'High noise')
f_tpr_tt.text(0.5, 0.015, 'Weight of the spatial distances', ha='center', 
                  fontsize=30)
ax_tpr_tt[0].set_ylabel(r'p-value (TPR)', fontsize=30)
lgd = ax_tpr_tt[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_tpr_tt.set_size_inches(20, 8)
plt.tight_layout(pad=1.5)

if do_save_fig:
    f_tpr_tt.savefig(op.join(path_fig, 'tpr_ttest.png'))


# In[]: Plots subjectwise
for idx_s, subject in enumerate(subjects):
    f_xi1_tt, ax_xi1_tt = plt.subplots(1, 2)
    for idx_a, alpha in enumerate(alpha_bio_tot):
        for idx_k, knn in enumerate(knn_tot):
            ax_xi1_tt[idx_a].plot(gamma_tot, 
                                  xi2_p_values[idx_k, :, idx_s, idx_a], 
                                  label='$k$ = %d'%knn, color=colors[idx_k])
            ax_xi1_tt[idx_a].set_yscale('log')
        ax_xi1_tt[idx_a].plot(gamma_tot, 
                              0.05*np.ones(np.size(gamma_tot)), 
                              'k--', label='p = 0.05')
    ax_xi1_tt[0].set_title(r'Low noise')
    ax_xi1_tt[1].set_title(r'High noise')
    f_xi1_tt.text(0.5, 0.015, 'Weight of the spatial distances', ha='center', 
                  fontsize=30)
    ax_xi1_tt[0].set_ylabel(r'p-value', fontsize=30)
    lgd = ax_xi1[1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
    f_xi1_tt.set_size_inches(20, 8)
    plt.tight_layout(pad=1.5)

    if do_save_fig:
        f_xi1_tt.savefig(op.join(path_fig, 'subs_ttest_{}.png'.format(subject)))


# In[]: Plot 5. AUC analysis with ic and coh
f_auc, ax_auc = plt.subplots(2, 2)
for idx_a, alpha in enumerate(alpha_bio_tot):
    for idx_k, knn in enumerate(knn_tot):
        ax_auc[idx_a, 0].errorbar(gamma_tot, 
               auc_ic_mean[idx_k, :, idx_a], 
               yerr=auc_ic_sem[idx_k, :, idx_a], label='$k$ = %d'%knn,
               color=colors[idx_k])
        ax_auc[idx_a, 1].errorbar(gamma_tot, 
               auc_coh_mean[idx_k, :, idx_a], 
               yerr=auc_coh_sem[idx_k, :, idx_a], label='$k$ = %d'%knn,
               color=colors[idx_k])
    
    ax_auc[idx_a, 0].errorbar(gamma_tot, 
           auc_ic_an_mean[idx_a]*np.ones(np.size(gamma_tot)), 
           yerr=auc_ic_an_sem[idx_a]*np.ones(np.size(gamma_tot)), 
           fmt='k--', label='DK')    
    ax_auc[idx_a, 1].errorbar(gamma_tot, 
           auc_coh_an_mean[idx_a]*np.ones(np.size(gamma_tot)), 
           yerr=auc_coh_an_sem[idx_a]*np.ones(np.size(gamma_tot)), 
           fmt='k--', label='DK')    
    
    ax_auc[idx_a, 0].set_ylim(0.6, 1)
    ax_auc[idx_a, 1].set_ylim(0.6, 1)
    ax_auc[idx_a, 0].set_ylabel(r'AUC $\beta^b = %1.2f$'%alpha)
       
ax_auc[0, 0].set_title('IC')
ax_auc[0, 1].set_title('COH')

    
lgd = ax_auc[0, -1].legend(bbox_to_anchor=(1, 0.5), loc='center left',
                borderaxespad=0.2)
f_auc.text(0.5, 0.015, 'Weight of the spatial distances', ha='center', 
                  fontsize=30)
f_auc.set_size_inches(20, 18)
plt.tight_layout(pad=2)
if do_save_fig:
    f_auc.savefig(op.join(path_fig, 'auc_ic_coh.png'))