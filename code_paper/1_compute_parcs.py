# %%
"""
Generate MEG informed cortical parcellations
============================================
"""


import os
import os.path as op
import numpy as np
import pickle

from mne import (read_forward_solution, pick_types_forward,
                 convert_forward_solution, read_labels_from_annot)

from megicparc import compute_distance_matrix, compute_parcellation

target = '../data'
subjects = ['k1_T1', 'k2_T1', 'k4_T1', 'k6_T1', 'k7_T1',
           'CC110045', 'CC110182', 'CC110174', 'CC110126', 'CC110056']

subjects_dir = op.join(target, 'subjects_flame')
sensors_meg = 'grad'
string_lf = op.join(target, 'fwd_models', 'original_fwd',
                  '{:s}_meg_single_layer-fwd.fif')

gamma_tot = np.arange(0, 1.01, 0.2)
knn_tot = [10, 20, 30, 40]
theta = 0.05
parc = 'aparc'

folder_fl = op.join(target, 'results', 'parcellations')
string_target_file = op.join(folder_fl,
                        '{:s}_flame_grad_k{:d}_gamma{:1.2f}_theta{:1.2f}.pkl')

if not op.isdir(folder_fl):
    os.mkdir(folder_fl)

# In[] Initialization
for idx_sub, subject in enumerate(subjects):
    print('Working with %s' % subject)

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
    for gamma in gamma_tot:
        sort_dist = compute_distance_matrix(fwd, gamma=gamma,
                                            theta=theta, labels=label)
        for knn in knn_tot:
            target_file = string_target_file.format(
                subject, knn, gamma, theta)
            flame_data = compute_parcellation(sort_dist, k_nn=knn)
            # - Save
            print('Saving %s' % target_file)
            aux_f = open(target_file, 'wb')
            pickle.dump(flame_data, aux_f, protocol=2)
            aux_f.close()

            del flame_data
        del sort_dist

    if subject == 'k2_T1' or subject == 'CC110182':  # For 'k2_T1' we also compute parcellations for theta = 0

        theta_aux = 0

        for gamma in gamma_tot:
            sort_dist = compute_distance_matrix(fwd, gamma=gamma,
                                                theta=theta_aux, labels=label)
            for knn in knn_tot:
                target_file = string_target_file.format(
                    subject, knn, gamma, theta_aux)
                flame_data = compute_parcellation(sort_dist, k_nn=knn)
                # - Save
                print('Saving %s' % target_file)
                aux_f = open(target_file, 'wb')
                pickle.dump(flame_data, aux_f, protocol=2)
                aux_f.close()

                del flame_data
            del sort_dist






# %%
