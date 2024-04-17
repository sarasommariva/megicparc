# Author: Sara Sommariva <sara.sommariva@aalto.fi>

import os.path as op
import numpy as np
import scipy.io

from mayavi import mlab

from mne import (read_forward_solution, pick_types_forward, 
                 convert_forward_solution, 
                 read_labels_from_annot, write_labels_to_annot)

import flame

# File in which the number of nearest neighbours has been updated in at least
# two points.
# '/m/nbe/work/sommars1/FLAME/flame_parcellations/k2_T1_cps_grad_an_method_1_gamma1_0.95/flame_grad_knn40_44_cc.mat'

# In[]: Step 1. Define general parameters
target = '/m/nbe/work/sommars1/FLAME/'
subject = 'k2_T1'
subjects_dir = op.join(target, 'subjects_flame')
theta = 0.05
sensors_meg = 'grad' # True or'grad'

wc = 11
k_nn = 40

do_plot = False
do_save = False

# NOTA: da matlab non ho prove per:
#      - magnetometri
#      - entrambi i sensori piu' le anatomical constrains.

if sensors_meg == True:
    aux_type = 'all'
elif sensors_meg == 'grad':
    aux_type = 'grad'

folder_fl = op.join(target, 'flame_parcellations', 
        '%s_cps_%s_an_method_1_gamma1_%1.2f'%(subject, aux_type, 1-theta))

# In[]: Step 2. Load forward solution  
path_lf = op.join(target, 'data', 
                  '%s_meg_single_layer-fwd.fif'%subject)
fwd = read_forward_solution(path_lf)
fwd = pick_types_forward(fwd, meg=sensors_meg, 
                eeg=False, ref_meg=False, exclude='bads')
fwd = convert_forward_solution(fwd, 
                surf_ori=True, force_fixed=True, use_cps=True)

# In[]: Step 3. Load antomical regions
# TODO: rendi questa parte eseguita solo su richiesta
parc = 'aparc' # 'aparc.a2009s'
label_lh = read_labels_from_annot(subject=subject, parc=parc, hemi='lh',
                           subjects_dir=subjects_dir)
label_rh = read_labels_from_annot(subject=subject, parc=parc, hemi='rh', 
                        subjects_dir=subjects_dir)
label = label_lh + label_rh

# In[]: Step 4. Run flame
gamma_tot = 1-np.arange(0, 1.01, 0.1)
gamma = gamma_tot[wc-1]

sort_dist = flame.compute_distance_matrix(fwd, gamma=gamma, 
                                          theta=theta, labels=label)

# In[]
#[density, nn_counts] =  flame.compute_density(sort_dist['vals'], k_nn)

# In[]

flame_data = flame.compute_parcellation(sort_dist, k_nn=k_nn)

# In[]:

# TODO: controllare che unendo tutte le regioni io riottenga il src completo.
flame_labels = flame.store_flame_labels(flame_data, fwd['src'], subject)

# In[]:       
if do_save: 
    write_labels_to_annot(flame_labels, subject=subject, 
                      parc='flame_k%d_gamma%1.1f'%(k_nn, gamma), 
                      overwrite=False, subjects_dir=subjects_dir, 
                      hemi='both')
    
# In[]: Step 5. Some checks
    
# In[]: Step 5.1. Comparison with .mat file
# TODO: Spostare dentro al codice? Qualche test anche su outliers?
#       Cosa mi direbbe la matematica?
n_rois = flame_data['centroids']
n_rois_warn = 0
rois_warn = []
for ir in range(n_rois):
    if not flame_data['centroids_id'][ir] in flame_data['parcel'][ir]:
        n_rois_warn += 1
        rois_warn.append(ir)

print(str(n_rois_warn) + ' parcels do not contain their centroids:')
print(rois_warn)

# TODO: Fare un controllo piu' furbo? 
path_flame = op.join(folder_fl,  
                    'flame_' + aux_type + '_knn' + str(k_nn) + '_' \
                    + str(wc) + str(wc) + '_cc.mat')
print('Loading ' + path_flame)
flame_temp = scipy.io.loadmat(path_flame)
flame_mat = flame_temp[scipy.io.whosmat(path_flame)[0][0]][0,0]
# ******    Notice: flame data now are in matlab indeces
nm = flame_mat['centroids'][0][0]
flame_mat['centroids_id'] = flame_mat['centroids_id'] - 1
for im in range(len(flame_mat['parcel'])):
    flame_mat['parcel'][im][0] = flame_mat['parcel'][im][0]-1
flame_mat['outliers_id'] = flame_mat['outliers_id'] - 1
    
# Check (this way I'm checking also the order) 
# 1. Number of region
assert flame_data['centroids'] == flame_mat['centroids'][0][0]
# 2. Number of outliers
assert flame_data['outliers'] == flame_mat['outliers'][0][0] 
# 3. Centroids
#assert flame_mat['centroids_id'].tolist()[0] == flame_data['centroids_id']
assert np.all(flame_data['centroids_id'] == flame_mat['centroids_id'].flatten())
# 4. Parcellations
roi_ord = []
for ir in range(flame_data['centroids']):
    test = False
    if flame_data['parcel'][ir].tolist() == flame_mat['parcel'][ir][0].T.tolist()[0]:
        test = True
    elif set(flame_data['parcel'][ir].tolist()) == set(flame_mat['parcel'][ir][0].T.tolist()[0]):
        test = True
        roi_ord.append(ir)
print('Rois with different order: ' + str(roi_ord))
# 5. Outliers
#assert set(flame_data['outliers_id']) == set(flame_mat['outliers_id'].flatten().tolist())
assert np.all(flame_data['outliers_id'] == flame_mat['outliers_id'].flatten())

# In[]: Step 5.2. Test that the labels cover the whole source-space
src = fwd['src']

vert_lh = [label.vertices.tolist() for label in flame_labels
                       if label.hemi == 'lh' and label.name != 'unknown-lh']
#vert_rh = [label.vertices.tolist() for label in flame_labels 
#                       if label.hemi=='rh' and label.name != 'unknown-rh']
vert_rh = [label.vertices.tolist() for label in flame_labels 
                       if label.hemi=='rh' and label.name != 'unknown-rh']

from itertools import chain
assert set(list(chain.from_iterable(vert_lh))) == set(src[0]['vertno'])
assert set(list(chain.from_iterable(vert_rh))) == set(src[1]['vertno']) 

# In[]: Step 5.3. Test on function for connected components

# TODO: tentare anche qualche plot? 
bmesh_adj_mat = flame.triangulation2adjacency(fwd['src'])
assert np.all(bmesh_adj_mat == bmesh_adj_mat.T)

for ir in range(flame_data['centroids']):
    conn_comp = flame.compute_connected_components(bmesh_adj_mat, 
                                               flame_data['parcel'][ir])
    concatenated_conn_comp = np.concatenate(conn_comp, axis=None)
    assert set(flame_data['parcel'][ir].tolist()) == set(concatenated_conn_comp.tolist())

# In[]:
if do_plot:
    idx = [1, 3, 20, 40, 500]
    
    brain = flame.plot_flame_labels(idx, flame_labels, src, 
                                    subject=subject, subjects_dir=subjects_dir, 
                                    plot_type='both')
            
    brain.add_foci(src[0]['vertno'], coords_as_verts=True, hemi='lh', color='yellow',
                   scale_factor=0.1, alpha=0.5)
    brain.add_foci(src[1]['vertno'], coords_as_verts=True, hemi='rh', color='yellow',
                   scale_factor=0.1, alpha=0.5)
                   
    mlab.show()   