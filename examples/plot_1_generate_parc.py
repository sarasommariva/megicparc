"""
Create a MEG informed cortical parcellation
===========================================
This example shows how to generate and visualize a lead-field based
MEG informed cortical parcellation.
"""
######################################################################
# Import the required

from megicparc import (compute_distance_matrix, compute_parcellation,
                       store_flame_labels)
from megicparc.viz import (plot_flame_labels, plot_flame_centroids)

from mne import (read_forward_solution, read_source_spaces,
                 read_labels_from_annot, pick_types_forward,
                 convert_forward_solution)
from mne.datasets import sample

import os.path as op

######################################################################
# Define input parameters for the flame algorithm running in megicperc

parc = 'aparc'
gamma = 0.8
theta = 0.05
sensors_meg = 'grad'
knn = 30

######################################################################
# Load lead-field matrix and source-space
data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'sample'
fwd_file = op.join(data_path, 'MEG', subject, 'sample_audvis-meg-eeg-oct-6-fwd.fif')
src_file = op.join('..', 'data', 'data_mne_sample', 'source_space_distance-src.fif')

fwd = read_forward_solution(fwd_file)
fwd = pick_types_forward(fwd, meg=sensors_meg, eeg=False,
                         ref_meg=False, exclude='bads')
fwd = convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                               use_cps=True)
src = read_source_spaces(src_file)
# Inter-source distances along the cortical surface has been added
# to the source-space as follows
#from mne import add_source_space_distances
#src = add_source_space_distances(fwd['src'])
#src.save(op.join('..', 'data', 'data_mne_sample', 'source_space_distance-src.fif'))

fwd['src'] = src

######################################################################
# Load the cortical atlas and run flame algorithm
label_lh = read_labels_from_annot(subject=subject, parc=parc, hemi='lh',
                                  subjects_dir=subjects_dir)
label_rh = read_labels_from_annot(subject=subject, parc=parc, hemi='rh',
                                  subjects_dir=subjects_dir)
label = label_lh + label_rh
sort_dist = compute_distance_matrix(fwd, gamma=gamma,
                                    theta=theta, labels=label)
sample_parc = compute_parcellation(sort_dist, k_nn=knn)
# Store megic parcels as mne-python labels for visualization purpose.
sample_parc_labels = store_flame_labels(sample_parc, src, subject)

""
plot_flame_centroids(sample_parc, fwd['src'], subject, subjects_dir,
                     brain=None, surf='inflated', scale_factor=0.5,
                     color='white')


""
brain_parc = plot_flame_labels([87], sample_parc_labels, src, subject,
                  subjects_dir, surf='inflated', brain=None,
                  color=None, plot_region=True,
                  plot_points=False, plot_borders=False)
plot_flame_labels([80], sample_parc_labels, src, subject,
                  subjects_dir, surf='inflated', brain=brain_parc,
                  color=None, plot_region=False,
                  plot_points=False, plot_borders=True)
plot_flame_labels([52], sample_parc_labels, src, subject,
                  subjects_dir, surf='inflated', brain=brain_parc,
                  color=None, plot_region=False,
                  plot_points=True, plot_borders=False)


""
# sphinx_gallery_thumbnail_number = 2
