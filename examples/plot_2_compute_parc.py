"""
Create multiple MEG informed cortical parcellations for further analysis
=======================================================================
Compute the lead-field based MEG informed cortical parcellations that will
be used to investigate the test the proposed method.
"""

######################################################################
# Import the required packages

import os
import os.path as op
import numpy as np
import pickle

from mne import (read_forward_solution, pick_types_forward,
                 convert_forward_solution, read_source_spaces,
                 read_labels_from_annot)
from mne.datasets import sample

from megicparc import compute_distance_matrix, compute_parcellation

######################################################################
# Define input parameters for the flame algorithm running in megicperc

gamma_tot = np.arange(0, 1.01, 0.2)
knn_tot = [10, 20, 30, 40]
theta = 0.05
parc = 'aparc'
sensors_meg = 'grad'

folder_fl = op.join('..', 'data', 'data_mne_sample')
string_target_file = op.join(folder_fl,
                        '{:s}_flame_grad_k{:d}_gamma{:1.2f}_theta{:1.2f}.pkl')

######################################################################
# Load lead-field matrix and source-space

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'sample'
fwd_file = op.join(data_path, 'MEG', subject, 'sample_audvis-meg-eeg-oct-6-fwd.fif')
src_file = op.join(folder_fl, 'source_space_distance-src.fif')

fwd = read_forward_solution(fwd_file)
fwd = pick_types_forward(fwd, meg=sensors_meg, eeg=False,
                         ref_meg=False, exclude='bads')
fwd = convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                               use_cps=True)
src = read_source_spaces(src_file)
fwd['src'] = src

######################################################################
# Load the cortical atlas

label_lh = read_labels_from_annot(subject=subject, parc=parc, hemi='lh',
                                  subjects_dir=subjects_dir)
label_rh = read_labels_from_annot(subject=subject, parc=parc, hemi='rh',
                                  subjects_dir=subjects_dir)
label = label_lh + label_rh

######################################################################
# Compute and save MEG-informed parcellations

for gamma in gamma_tot:
    sort_dist = compute_distance_matrix(fwd, gamma=gamma,
                                        theta=theta, labels=label)
    for knn in knn_tot:

        target_file = string_target_file.format(
            subject, knn, gamma, theta)

        if op.exists(target_file):
            print('The following file already exists: %s' %target_file)
        else:
            flame_data = compute_parcellation(sort_dist, k_nn=knn)
            # - Save
            print('Saving %s' % target_file)
            aux_f = open(target_file, 'wb')
            pickle.dump(flame_data, aux_f, protocol=2)
            aux_f.close()

            del flame_data
    del sort_dist

######################################################################
# Compute and save MEG-informed parcellations for theta=0 and
# k = 30

theta_aux = 0
knn_aux = 30

for gamma in gamma_tot:
    sort_dist = compute_distance_matrix(fwd, gamma=gamma,
                                        theta=theta_aux, labels=label)
    target_file = string_target_file.format(
        subject, knn_aux, gamma, theta_aux)
    if op.exists(target_file):
            print('The following file already exists: %s' %target_file)
    else:
        flame_data = compute_parcellation(sort_dist, k_nn=knn_aux)
        # - Save
        print('Saving %s' % target_file)
        aux_f = open(target_file, 'wb')
        pickle.dump(flame_data, aux_f, protocol=2)
        aux_f.close()
        del flame_data
    del sort_dist

""

