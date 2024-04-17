# Author: Sara Sommariva <sara.sommariva@aalto.fi>

# TODO: controllare di non far casino nel copiare le cose/gli array (python non matlab!!!)
# TODO: end comments
# TODO: Sanity check
# TODO: what sanity check for the forward model? (type of channels, cortico-cortical constrain)
# TODO: come posso salvare le mie parcellizzazioni? Ha senso usare le classi label di mne-python?

from __future__ import division, print_function
import numpy as np
import scipy
import scipy.linalg

from mne import Label
from mne.io.pick import channel_type
from mne.label import _n_colors


def compute_cosine_similarity(fwd):
    """
    Define cosine similarity s_ij^c
    
    Parameters:
    -----------
    fwd : instance of Forward
        The forward solution
        
    Returns:
    --------
    cosine_similarity : ndarray of float, shape (n_sources, n_sources)
        Similarity matrix s_ij^c
        
    Notes:
    ------
    For each pair of points in the source-space, i, j = 1, ..., n, 
            s_ij^c = | 1/S sum_{s=1}^S s_ij^c(s) |
    where S is the number of sensor types present in the forward model, and
    s_ij^c(s) is the cosine similarity between i-th and j-th columns
    of the leadfield for sensor type s.
    """
    
    # Step 1. Understand what sensor type are present in the fwd solution
    info = fwd['info']
    picks = np.arange(0, info['nchan'])
    types = np.array([channel_type(info, idx) for idx in picks])
    ch_types_used = set(types)  

    # Step 2. Exctract and normalized leadfield for each sensor type
    aux_L = fwd['sol']['data']
    L_norm = np.zeros(aux_L.shape)
    for this_type in ch_types_used:
        print('Normalizing leadfield for %s sensors'%this_type)
        idx = picks[types == this_type]
        Ns = idx.shape[0]
        L_this_type = aux_L[idx]
        L_this_type = L_this_type - np.mean(L_this_type, axis=0)
        L_norm[idx] = L_this_type / \
                      np.sqrt(1/(Ns-1)*np.sum(L_this_type**2, axis=0))
    
    # Step 3. Compute cosine similarity matrix
    cosine_similarity = abs(np.corrcoef(L_norm.T))
    
    return cosine_similarity

def compute_anatomical_mask(labels, src, constr_out=True):
    """
    Define anatomical constraints d_ij^a
    
    Parameters:
    -----------
    labels : list of Label
        Anatomical region used as reference
    src : SourceSpaces
        Source space to be parcellated
    constr_out : bool
        Wheter or not use anatomical constraints also on outliers (*).
        
    Reutrns:
    --------
    anat_mask : ndarray of float, shape (n_sources, n_sources)
        Distance matrix defining the anatomical constrains
        
    Notes.
    -----
    d_ij^a = 0 if vi and vj are in the same anatomical region 
                  or they are both outlier within the same hemi (*)
    d_ij^a = 1 otherwise
    """
    
    # TODO: this function seems to work only if src include both hemis
    
    n_vert = src[0]['nuse'] + src[1]['nuse']
    anat_mask = np.ones([n_vert, n_vert])
    visited = np.array([]).astype('int64')
    
    for ir in range(len(labels)):
        # Project labels on the decimated src
        # NB: It seems that I cannot use label.get_vertices_used because it 
        # returns points, while I need the indeces
        if labels[ir].hemi == 'lh':
            idx_src = src[0]['vertno']
            comp = 0
        elif labels[ir].hemi == 'rh':
            idx_src = src[1]['vertno']
            comp = src[0]['nuse']
        idx_roi = np.where(np.in1d(idx_src, labels[ir].vertices))[0] + comp
        anat_mask[np.ix_(idx_roi, idx_roi)] = 0
        visited = np.concatenate((visited, idx_roi))
        
    if constr_out:    
        out_lh = np.where(
                np.in1d(np.arange(0, src[0]['nuse']), visited)==False)[0] 
        out_rh = np.where(
                np.in1d(np.arange(0, src[1]['nuse'])+src[0]['nuse'], 
                        visited) == False)[0] + src[0]['nuse']    
        anat_mask[np.ix_(out_lh, out_lh)] = 0
        anat_mask[np.ix_(out_rh, out_rh)] = 0

    return anat_mask

def read_cortico_similarity(src):
    """
    Read cortico-cortical similarity matrix from source space
    
    Parameters:
    -----------
    src : SourceSpaces
        Source space from which the cortico-cortical similarity has to be read
    
    Returns:
    --------
    cortico_similarity : ndarray of float, shape (n_sources, n_sources)
        Cortico-cortical similarity matrix
    """
    
    # TODO: aggiungere un controllo che la distanza cortico-corticale sia 
    #       stata calcolate propriamente. Per esempio nel sample-subject di 
    #       MNE-python e' 0 per la maggior parte dei soggetti.
    # Step 1. Read distance from each hemi
    tmp_Dl = src[0]['dist'][np.ix_(src[0]['vertno'], src[0]['vertno'])]
    Dl = tmp_Dl.A
    tmp_Dr = src[1]['dist'][np.ix_(src[1]['vertno'], src[1]['vertno'])]
    Dr = tmp_Dr.A
    
    # Step 2. Normalize
    max_dist = max(np.max(Dl), np.max(Dr))
    Dl = Dl / max_dist
    Dr = Dr / max_dist
    
    # Trick to prevent rounding error for points in different hemispheres
    Sl = 1 - Dl 
    Sr = 1 - Dr   
    cortico_similarity = scipy.linalg.block_diag(Sl, Sr)
    
    return cortico_similarity

def compute_distance_matrix(fwd, gamma=0, theta=0, labels=None):
    
    """Compute combined distance matrix and sort it.
    
    Parameters
    ----------
    fwd : instance of Forward
        The forward solution 
    gamma : float in [0, 1]
        Weight given to the spatial constraints. 
    theta : float in [0, 1) 
        Weight give to the anatomical constraints. If None only the 
        cortico-cortical distance is used
    labels : list of Label
        Anatomical region used as reference
        
    Returns 
    -------
    sort_dist : dict
        A dictionary containing:
        sort_dist['vals'] : ndarray, shape (n_sources, n_sources-1)
        Distance matrix, each row sorted in ascending order.
        sort_dist['idx'] : ndarray, shape (n_source, n_sources-1)
        Indices that would sort the distance matrix
    """
 
    if gamma < 0 or gamma > 1:
        raise ValueError("gamma must be in [0, 1]")
    if theta <0 or theta >= 1:
        raise ValueError("theta must be in [0, 1)")
    elif theta > 0 and labels==None:
        raise ValueError("Anatomical labels needs to be provided")
        
    print('Using leadfield of size = ' + str(fwd['sol']['data'].shape))
    print('Computing distance matrix for gamma=%1.2f theta=%1.2f'
          %(gamma, theta))
    
    src = fwd['src']
    n_vert = src[0]['nuse'] + src[1]['nuse']
    
    # 1. Cortico cortical similarity matrix
    # TODO: add control that cosine and cortico similarity are computed only
    #       if necessary
    # TODO: casino se theta=0 e labels=None
    if gamma > 0:
        
        print('Reading cortical distance from src')
        cortico_similarity = read_cortico_similarity(src)
    
        if theta > 0:
            
            print('Adding anatomical constraints')
            an_mask = compute_anatomical_mask(labels, src)
            cortico_similarity = (1-theta) * cortico_similarity + \
                                theta * (1-an_mask)
    else:
        cortico_similarity = np.zeros([n_vert, n_vert])
        
    # 2. Compute cosine distance
    if gamma < 1:
        
        print('Computing cosine distance')
        cosine_similarity = compute_cosine_similarity(fwd)
        
    else:
        
        cosine_similarity = np.zeros([n_vert, n_vert])
        
    # 3. Compute combined distance
    similarity_matrix = (1-gamma) * cosine_similarity + \
                    gamma * cortico_similarity 
    distance_matrix = 1 - similarity_matrix
    
    # Let's meake some space
    del cosine_similarity, cortico_similarity
    
    # Sort combined distance
    dist_idx = np.argsort(distance_matrix, axis=1)[:, 1:]
    dist_vals = np.array([distance_matrix[iv, dist_idx[iv]] \
                    for iv in range(distance_matrix.shape[0])])

    sort_dist = {'vals': dist_vals, 'idx' : dist_idx}
    
    return sort_dist

def compute_density(sort_dist_vals, k_nn):
    
    """Compute density function
                 pho(i) = 1/sum_{j in N(i)} d_ij, i = 1, ..., n
    
    Parameters 
    ----------
    sort_dist_vals : ndarray of float, shape (n_sources, n_sources-1)
        Distance matrix, each row sorted in ascending order.
    k_nn : int in (0, +inf]
        Number of nearest neighbours
        
    Returns
    -------
    density : array of float, shape (n_sources, )
        Density function
    nncounts : array of int, shape (n_sources, )
        Actual size of the sets of nearest neighbours
    """
    
    if k_nn <= 0:
        raise ValueError("Number of nearest neighbour k_nn must be positive")
        
    if not k_nn == int(k_nn):
        raise TypeError("Number of nearest neighbour k_nn must be an integer")
    else: 
        k_nn = int(k_nn) # k_nn will be used as index.
    
    # TODO: more clever k_max?
    k_max = 50
    n_vert = sort_dist_vals.shape[0]
    print('Check n_vert = ' + str(n_vert))
    
    nncounts = np.zeros(n_vert, dtype=int)
    density = np.zeros(n_vert, dtype=float)
    # TODO: Do I have a parcellations with nncounts changes with respect to k 
    #       to test this part?????
    #       Differenze di precisione con MATLAB????
    for iv in range(n_vert):
        nncounts[iv] = min(k_max, 
                1+np.where(
                sort_dist_vals[iv] == sort_dist_vals[iv, k_nn-1])[0][-1])
        density[iv] = 1 / (np.sum(sort_dist_vals[iv, 0:nncounts[iv]]) + 1e-9)
    
    return density, nncounts

def compute_parcellation(sort_dist, k_nn):
    
    """Run flame algorithm
    
    Parameters
    ----------
    sort_dist : dict
        A dictionary containing:
        sort_dist['vals'] : ndarray, shape (n_sources, n_sources-1)
        Distance matrix, each row sorted in ascending order.
        sort_dist['idx'] : ndarray, shape (n_source, n_sources-1)
        Indices that would sort the distance matrix
    k_nn : int in (0, +inf]
        Number of nearest neighbours
        
    Returns
    -------
    flame_data : dict
        Results of flame algorithm
        
    Notes
    -----
    flame_data contains:
    centroids : int
        Number of regions
    centroids_id : array of int, shape (n_roi, )
        Indices of the points that are centroids
    outliers : int
        Number of outliers
    outliers_id : array of int, shape (n_out, )
        Indices of the points that are outliers
    weights : ndarray of float, shape (n_vert, k_nn)
        Weights used in the computation of fuzzy membership
    nncounts : array of int, shape (n_vert, )
        Actual size of the sets of nearest neighbours
    parcel : list of array
        List of all the parcels. The last parcel contains the outliers (if any)
    
    """
    
    dist_vals = sort_dist['vals']
    dist_idx = sort_dist['idx']
    n_vert = dist_vals.shape[0]
    
    # In[]: Step 1. Define centroids and outliers
    # 1.1. Compute density 
    density, nncounts = compute_density(dist_vals, k_nn)
    
    # 1.2. Define thresholds for outliers
    th = -2
    sum_ = np.sum(density) / n_vert
    sum2_ = np.sum(density**2) / n_vert
    th = sum_ + th * np.sqrt(sum2_ - sum_ * sum_)
    
    # 1.3. Define outliers and centroids
    centroids = 0
    centroids_id = []
    outliers = 0
    outliers_id = []
    idd = []
    
    for iv in range(n_vert):
        k = nncounts[iv]
        fmax = 0
        fmin = density[iv] / density[dist_idx[iv, 0]]
        for ik in np.arange(1, k):
            aux = density[iv] / density[dist_idx[iv, ik]]
            if aux > fmax:
                fmax = aux
            if aux < fmin:
                fmin = aux
                
            # If vertex i has a neighbour that has been already classified
            # as centroids or outlier cannot be a centroid
            if dist_idx[iv, ik] in idd:
                fmin = 0
            
        if fmin >= 1.0:
            centroids += 1
            centroids_id.append(iv)
        elif fmax <= 1.0 and density[iv] < th:
            outliers += 1
            outliers_id.append(iv)
            
        idd = centroids_id + outliers_id
        
    # In[]: Step 2. Assign fuzzy memberships 
    
    # 2.1. Initialize membership and define weigths
    # TODO: more clevere way than a for???
    weights = np.zeros([n_vert, max(nncounts)])
    fmembership = np.zeros([n_vert, centroids+1])
    class_ = 0
    for iv in range(n_vert):
        if iv in centroids_id:
            fmembership[iv, class_] = 1.0
            class_ += 1
        elif iv in outliers_id:
            fmembership[iv, centroids] = 1.0
        else:
            fmembership[iv] = 1 / (centroids+1)
        
    # Weights
        k = nncounts[iv]
        sum_ = 0.5 * k * (k+1)
        weights[iv, 0:k] = np.arange(k, 0, -1) / sum_
        
            
    # 2.2. Optimization routine
    idx_ = [x for x in range(n_vert) if x not in centroids_id + outliers_id]
    k_nn_max = max(nncounts)
    res = np.zeros([k_nn_max*len(idx_), centroids+1])
    w = np.reshape(weights[idx_], [1, k_nn_max*len(idx_)])
    # NB: Python and matlab reshape seem to have different behaviour 
    
    fmembership_ = np.copy(fmembership)
    
    nstep_max = 1000
    nstep = 0
    tol = np.inf
    while (nstep < nstep_max) and (tol > 1e-6):
        for j in range(centroids+1):
            temp_res = np.multiply(w, 
                 fmembership[np.reshape(dist_idx[idx_, 0:k_nn_max], [1, k_nn_max*len(idx_)]), j])
            res[:, j] = temp_res
        F_upd = np.cumsum(res, axis=0)
        F_upd = F_upd[np.arange(k_nn_max, F_upd.shape[0]+1, k_nn_max)-1, :]
        F_upd[1:, :] = F_upd[1:, :] - F_upd[0:-1, :]
        fmembership_[idx_, :] = F_upd
        
        tol = np.linalg.norm(fmembership - fmembership_)**2
        np.copyto(fmembership, np.divide(fmembership_,
                 np.tile(np.sum(fmembership_, axis=1), [centroids+1, 1]).T))
        nstep += 1
        print(nstep)
        print(tol)
        
    # 2.3. Defuzzification (one-to-one)
    # 2.3.1. Update weights (also for centroids and outliers)
    # TODO: L'entropia sembra essere ordinata in modo crescente. E' quello che vogliamo?
    # TODO: Quelli che erano originariamente outliers possono diventare altre cose
    #       --> Per questo i miei outliers erano diversi!!!!!!
    
    del res, F_upd
    res = np.zeros([k_nn_max*n_vert, centroids+1])
    w = np.reshape(weights, [1, k_nn_max*n_vert])
    
    for j in range(centroids+1):
        temp_res = np.multiply(w, 
                 fmembership[np.reshape(dist_idx[:, 0:k_nn_max], [1, k_nn_max*n_vert]), j])
        res[:, j] = temp_res
    F_upd = np.cumsum(res, axis=0)
    F_upd = F_upd[np.arange(k_nn_max, F_upd.shape[0]+1, k_nn_max)-1, :]
    F_upd[1:, :] = F_upd[1:, :] - F_upd[0:-1, :]
    
    # 2.3.2 Sort by entropy
    fm_e = np.zeros(n_vert)
    for iv in range(n_vert):
        fm_e[iv] = np.sum(-np.multiply(F_upd[iv], np.log(F_upd[iv])))
        
    idx_e = np.argsort(fm_e)
    F_upd_e = F_upd[idx_e]
    
    # 2.3.3. Defuzzification
    parcel = []
    crisp_memb = np.argmax(F_upd_e, axis=1)
    for ir in range(centroids+1):
        parcel.append(idx_e[crisp_memb == ir])
    
    # 2.4. Some sanity check
    ir = 0
    empty_region = 0
    deleted_centr = 0
    new_out = []
    while ir < centroids:
        # -- Delete empty regions
        if parcel[ir].shape[0] == 0:
            del centroids_id[ir]
            del parcel[ir]                
            centroids -= 1
            empty_region += 1
        # -- Consider outliers points in regions without their centroids
        elif not centroids_id[ir] in parcel[ir]:
            deleted_centr += 1
            new_out = new_out + parcel[ir].tolist()
            del centroids_id[ir] 
            del parcel[ir]
            centroids -= 1
        else:
            ir += 1
    
    if empty_region > 0:
        print('%d empty region(s) deleted'%empty_region)
    
    if deleted_centr > 0:
        print('Deleted %d region(s) without centroid'%deleted_centr)
        print('Added %d outliers'%len(new_out))
        parcel[centroids] = np.concatenate(
                    (parcel[centroids], np.array(new_out)), axis=0)
    
    
    # -- outliers
    # TODO: Store outliers in a better way?? 
    if parcel[centroids].shape[0] == 0:
        del parcel[centroids]
        outliers = 0
        outliers_id = []
    else:
        outliers = parcel[centroids].shape[0]
        outliers_id = parcel[centroids].tolist()
        
    # TODO: Are line 218-220 necesary?????


# TODO: Store also the number of iterations? 

    flame_data = {'centroids' : centroids, 'outliers' : outliers,
                  'centroids_id' : np.asarray(centroids_id),  
                  'outliers_id' : np.asarray(outliers_id),
                  'weights' : weights, 'nncounts' : nncounts,
                  'parcel' : parcel}
    
    return flame_data

def compute_parcellation_TEST(sort_dist, k_nn):
    
    """Run flame algorithm
    
    Parameters
    ----------
    sort_dist : dict
        A dictionary containing:
        sort_dist['vals'] : ndarray, shape (n_sources, n_sources-1)
        Distance matrix, each row sorted in ascending order.
        sort_dist['idx'] : ndarray, shape (n_source, n_sources-1)
        Indices that would sort the distance matrix
    k_nn : int in (0, +inf]
        Number of nearest neighbours
        
    Returns
    -------
    flame_data : dict
        Results of flame algorithm
        
    Notes
    -----
    flame_data contains:
    centroids : int
        Number of regions
    centroids_id : array of int, shape (n_roi, )
        Indices of the points that are centroids
    outliers : int
        Number of outliers
    outliers_id : array of int, shape (n_out, )
        Indices of the points that are outliers
    weights : ndarray of float, shape (n_vert, k_nn)
        Weights used in the computation of fuzzy membership
    nncounts : array of int, shape (n_vert, )
        Actual size of the sets of nearest neighbours
    parcel : list of array
        List of all the parcels. The last parcel contains the outliers (if any)
    
    """
    
    dist_vals = sort_dist['vals']
    dist_idx = sort_dist['idx']
    n_vert = dist_vals.shape[0]
    
    # In[]: Step 1. Define centroids and outliers
    # 1.1. Compute density 
    density, nncounts = compute_density(dist_vals, k_nn)
    
    # 1.2. Define thresholds for outliers
    th = -2
    sum_ = np.sum(density) / n_vert
    sum2_ = np.sum(density**2) / n_vert
    th = sum_ + th * np.sqrt(sum2_ - sum_ * sum_)
    
    # 1.3. Define outliers and centroids
    centroids = 0
    centroids_id = []
    outliers = 0
    outliers_id = []
    idd = []
    
    for iv in range(n_vert):
        k = nncounts[iv]
        fmax = 0
        fmin = density[iv] / density[dist_idx[iv, 0]]
        for ik in np.arange(1, k):
            aux = density[iv] / density[dist_idx[iv, ik]]
            if aux > fmax:
                fmax = aux
            if aux < fmin:
                fmin = aux
                
            # If vertex i has a neighbour that has been already classified
            # as centroids or outlier cannot be a centroid
            if dist_idx[iv, ik] in idd:
                fmin = 0
            
        if fmin >= 1.0:
            centroids += 1
            centroids_id.append(iv)
        elif fmax <= 1.0 and density[iv] < th:
            outliers += 1
            outliers_id.append(iv)
            
        idd = centroids_id + outliers_id
        
    # In[]: Step 2. Assign fuzzy memberships 
    
    # 2.1. Initialize membership and define weigths
    # TODO: more clevere way than a for???
    weights = np.zeros([n_vert, max(nncounts)])
    fmembership = np.zeros([n_vert, centroids+1])
    class_ = 0
    for iv in range(n_vert):
        if iv in centroids_id:
            fmembership[iv, class_] = 1.0
            class_ += 1
        elif iv in outliers_id:
            fmembership[iv, centroids] = 1.0
        else:
            fmembership[iv] = 1 / (centroids+1)
        
    # Weights
        k = nncounts[iv]
        sum_ = 0.5 * k * (k+1)
        weights[iv, 0:k] = np.arange(k, 0, -1) / sum_
        
            
    # 2.2. Optimization routine
    idx_ = [x for x in range(n_vert) if x not in centroids_id + outliers_id]
    k_nn_max = max(nncounts)
    res = np.zeros([k_nn_max*len(idx_), centroids+1])
    w = np.reshape(weights[idx_], [1, k_nn_max*len(idx_)])
    # NB: Python and matlab reshape seem to have different behaviour 
    
    fmembership_ = np.copy(fmembership)
    
    nstep_max = 1000
    nstep = 0
    tol = np.inf
    while (nstep < nstep_max) and (tol > 1e-6):
        for j in range(centroids+1):
            temp_res = np.multiply(w, 
                 fmembership[np.reshape(dist_idx[idx_, 0:k_nn_max], [1, k_nn_max*len(idx_)]), j])
            res[:, j] = temp_res
        F_upd = np.cumsum(res, axis=0)
        F_upd = F_upd[np.arange(k_nn_max, F_upd.shape[0]+1, k_nn_max)-1, :]
        F_upd[1:, :] = F_upd[1:, :] - F_upd[0:-1, :]
        fmembership_[idx_, :] = F_upd
        
        tol = np.linalg.norm(fmembership - fmembership_)**2
        np.copyto(fmembership, np.divide(fmembership_,
                 np.tile(np.sum(fmembership_, axis=1), [centroids+1, 1]).T))
        nstep += 1
        print(nstep)
        print(tol)
        
    # 2.3. Defuzzification (one-to-one)
    # 2.3.1. Update weights (also for centroids and outliers)
    # TODO: L'entropia sembra essere ordinata in modo crescente. E' quello che vogliamo?
    # TODO: Quelli che erano originariamente outliers possono diventare altre cose
    #       --> Per questo i miei outliers erano diversi!!!!!!
    
    del res, F_upd
    res = np.zeros([k_nn_max*n_vert, centroids+1])
    w = np.reshape(weights, [1, k_nn_max*n_vert])
    
    for j in range(centroids+1):
        temp_res = np.multiply(w, 
                 fmembership[np.reshape(dist_idx[:, 0:k_nn_max], [1, k_nn_max*n_vert]), j])
        res[:, j] = temp_res
    F_upd = np.cumsum(res, axis=0)
    F_upd = F_upd[np.arange(k_nn_max, F_upd.shape[0]+1, k_nn_max)-1, :]
    F_upd[1:, :] = F_upd[1:, :] - F_upd[0:-1, :]
    
    # 2.3.2 Sort by entropy
    fm_e = np.zeros(n_vert)
    for iv in range(n_vert):
        fm_e[iv] = np.sum(-np.multiply(F_upd[iv], np.log(F_upd[iv])))
        
    idx_e = np.argsort(fm_e)
    F_upd_e = F_upd[idx_e]
    
    # 2.3.3. Defuzzification
    parcel = []
    crisp_memb = np.argmax(F_upd_e, axis=1)
    for ir in range(centroids+1):
        parcel.append(idx_e[crisp_memb == ir])
    
    # 2.4. Some sanity check
    ir = 0
    empty_region = 0
    deleted_centr = 0
    new_out = []
    while ir < centroids:
        # -- Delete empty regions
        if parcel[ir].shape[0] == 0:
            del centroids_id[ir]
            del parcel[ir]                
            centroids -= 1
            empty_region += 1
        # -- Consider outliers points in regions without their centroids
        elif not centroids_id[ir] in parcel[ir]:
            deleted_centr += 1
            new_out = new_out + parcel[ir].tolist()
            del centroids_id[ir] 
            del parcel[ir]
            centroids -= 1
        else:
            ir += 1
    
    if empty_region > 0:
        print('%d empty region(s) deleted'%empty_region)
    
    if deleted_centr > 0:
        print('Deleted %d region(s) without centroid'%deleted_centr)
        print('Added %d outliers'%len(new_out))
        parcel[centroids] = np.concatenate(
                    (parcel[centroids], np.array(new_out)), axis=0)
    
    
    # -- outliers
    # TODO: Store outliers in a better way?? 
    if parcel[centroids].shape[0] == 0:
        del parcel[centroids]
        outliers = 0
        outliers_id = []
    else:
        outliers = parcel[centroids].shape[0]
        outliers_id = parcel[centroids].tolist()
        
    # TODO: Are line 218-220 necesary?????


# TODO: Store also the number of iterations? 

    flame_data = {'centroids' : centroids, 'outliers' : outliers,
                  'centroids_id' : np.asarray(centroids_id),  
                  'outliers_id' : np.asarray(outliers_id),
                  'weights' : weights, 'nncounts' : nncounts,
                  'parcel' : parcel, 
                  'membership_e' : F_upd_e, 'membership' : F_upd}
    
    return flame_data

# TODO: pensare se sarebbe meglio inglobare il salvataggio come label fin da subito
    # Contro: flame_data contengono indici adatti al src ridotto
def store_flame_labels(flame_data, src, subject):
    """Store flame parcels as instances of Label
    
    Parameters
    ----------
    flame_data : dict
        Results of flame algorithm
    src : SourceSpaces
        Source space that has been parcellated
    subject : str
        Name of the subject
        
    Returns
    -------
    flame_labels : list of Labels
        Flame parcellation
    """
    
    nvert_lh = src[0]['nuse']
    flame_labels = []
    
    colors = _n_colors(flame_data['centroids']+2, 
                       bytes_=False, cmap='hsv')

    for ir in range(flame_data['centroids']):
        # Step 1. Understand to which hemi the region belong
        idx_roi = flame_data['parcel'][ir]
        idx_roi_lh = idx_roi[np.where(idx_roi < nvert_lh)[0]]
        idx_roi_rh = idx_roi[np.where(idx_roi >= nvert_lh)[0]] - nvert_lh
        if idx_roi_rh.size > 0 and idx_roi_lh.size > 0:
            # case 1. the region split over both hemis
            flame_labels.append(
                Label(np.sort(src[0]['vertno'][idx_roi_lh]), hemi='lh', 
                  comment='This region splits over both hemispheres', 
                  name='roi%d-lh'%(ir+1), subject=subject, color=colors[ir]))
            flame_labels.append(
                Label(np.sort(src[1]['vertno'][idx_roi_rh]), hemi='rh', 
                  comment='This region splits over both hemis', 
                  name='roi%d-rh'%(ir+1), subject=subject, color=colors[ir]))
        elif idx_roi_rh.size > 0:
            # case 2. righ hemi
            flame_labels.append(
                Label(np.sort(src[1]['vertno'][idx_roi_rh]), hemi='rh',  
                  name='roi%d-rh'%(ir+1), subject=subject, color=colors[ir]))
        elif idx_roi_lh.size > 0:
            # case 3. left hemi
            flame_labels.append(
                Label(np.sort(src[0]['vertno'][idx_roi_lh]), hemi='lh',  
                  name='roi%d-lh'%(ir+1), subject=subject, color=colors[ir]))  
                  
    # Outliers
    # TODO: qual'e' il modo ottimale per controllare se ci sono outliers? 
    if flame_data['outliers'] > 0:         
        idx_roi = flame_data['parcel'][-1]
        idx_roi_lh = idx_roi[np.where(idx_roi < nvert_lh)[0]]
        idx_roi_rh = idx_roi[np.where(idx_roi >= nvert_lh)[0]] - nvert_lh
        if idx_roi_rh.size > 0:
            # case 2. righ hemi
            flame_labels.append(
                Label(np.sort(src[1]['vertno'][idx_roi_rh]), hemi='rh',  
                      name='outliers-rh', subject=subject, color=colors[-1]))  
        if idx_roi_lh.size > 0:
            # case 3. left hemi
            flame_labels.append(
                Label(np.sort(src[0]['vertno'][idx_roi_lh]), hemi='lh',  
                      name='outliers-lh', subject=subject, color=colors[-2])) 

    return flame_labels

def triangulation2adjacency(src):
    """
    Compute the adjacency matrix defined by the source space mesh
    
    Parameters
    ----------
    src : SourceSpaces
        Source space for both right and left hemi containing the triangulation
        
    Returns
    -------
    adj_mat : ndarray of float, shape (n_sources, n_sources)
        Adjacency matrix
    """
    
    # 1.1. Project trinagulation on decimated source-space
    red_tris_lh = np.searchsorted(src[0]['vertno'], src[0]['use_tris']) 
    red_tris_rh = np.searchsorted(src[1]['vertno'], src[1]['use_tris']) 
    red_tris_rh = red_tris_rh + src[0]['nuse']
    red_all_tris = np.concatenate((red_tris_lh, red_tris_rh), axis=0)
    
    # 1.3. Construct adjacency matrix
    n_vert = src[0]['nuse'] + src[1]['nuse']
    adj_mat = np.zeros([n_vert, n_vert])
    for iv in range(3):
        for jv in np.arange(iv, 3):
            adj_mat[red_all_tris[:, iv], red_all_tris[:, jv]] = 1
            adj_mat[red_all_tris[:, jv], red_all_tris[:, iv]] = 1

    return adj_mat

def compute_connected_components(adj_mat, region):
    """
    Compute connected components for a given flame region
    
    Parameters
    ----------
    adj_mat : ndarray of float, shape (n_sources, n_sources)
        Adjacency matrix
    region : array of float, shape (n_v_reg, )
        Region to be considered
        
    Returns
    -------
    comps : list of array
        Set of connected components
    """
    
    n_comp = 0
    comps = []
    region_ = region.copy()
    
    while region_.shape[0] > 0:
        
        n_comp += 1
        visited = np.array([region_[0]])
        region_ = np.delete(region_, 0, axis=0)
        n_start = 0
        
        while region_.shape[0]>0 and n_start < visited.shape[0]:
            
            start = visited[n_start]
            neigh, _, index_neigh = np.intersect1d(np.where(adj_mat[start])[0], 
                                                 region_, return_indices=True)
            visited = np.append(visited, neigh)
            region_ = np.delete(region_, index_neigh)
            n_start += 1
            
        comps.append(visited)
    
    return comps

def compute_distinguishability_index(R_p, parcels, num_p):
    """Compute distinguishability index from a given parcel resolution matrix.
    
    Parameters
    ----------
    R_p : ndarray of float, shape (n_roi, n_vert)
        Parcel resolution matrix
    parcels : list of array
        Parcels. The last parcel contains the outliers if present
    num_p : int
        Number of parcels
        
    Returns:
    --------
    CT_mat : ndarray of float, shape (n_roi, n_roi)
        Crosstalk matrix
    DI : float
        Distinguishability index
    """
    
    if num_p != R_p.shape[0]:
        raise ValueError('Bad shape for the parcels resolution matrix')
    if (num_p != len(parcels) and num_p != len(parcels) - 1):
        raise ValueError('Bad number of provided regions')
        
    # Step 1. Square RM
    R_p2 = R_p ** 2
    
    # Step 2. Compute cross-talk matrix
    activity_tot = np.sum(R_p2, axis=1)   
    CT_mat = np.array(
            [np.divide(np.sum(R_p2[:, parcels[ip]], axis=1), activity_tot) 
            for ip in range(num_p)]).T
        
    # Step 3. Compute DI
    I = np.eye(num_p)
    demean_CT_mat = CT_mat - CT_mat.mean()
    demean_I = I - I.mean()
    DI = np.sum(demean_CT_mat * demean_I) / \
         np.sqrt(np.sum(demean_CT_mat**2)*np.sum(demean_I**2))
    
    return CT_mat, DI

def compute_localization_error(peaks, centroids_id, src):
    """
    Compute distance between vertices and the centroid in which they generate
    the highest activity
    
    Parameters:
    -----------
    peaks : ndarray of int, shape (n_vert, )
        Region (index) showing the highest activity
    centroids_id : ndarrya of int, shape (n_vert, )
        Indices of the points that are centroids
    src : SourceSpaces
        Source space
        
    Returns:
    --------
    loc_error : ndarray of float, shape (n_vert, )
        Euclidean localization error in mm
    """
    
    src_coord = np.concatenate(
                        (src[0]['rr'][src[0]['vertno']],
                         src[1]['rr'][src[1]['vertno']]), axis=0) 
    # TODO: are the mne-python src coordinates always in meter?              
    loc_error = np.linalg.norm(src_coord - src_coord[centroids_id[peaks]], 
                           ord=2, axis=1)*1000
    
    return loc_error

def membership2vector(parcel, n_vert):
    
    true_c = np.zeros(n_vert, dtype=np.int64) - 1
    for ir in range(len(parcel)):
        true_c[parcel[ir]] = ir
    
    return true_c


def labels_to_array(labels, src, type_index='python'):
    """Project MNE Label(s) to a given source-space and save as list of arrays
    
    Parameters:
    -----------
    labels : list of Label
        Label(s) to be projected
    src : SourceSpaces
        Source space 
    type_index : string ['python' | 'matlab']
    """
    
    name = [lab.name for lab in labels]
    parcels = list()
    total_vertx = np.ones(src[0]['nuse']+src[1]['nuse'])
    
    for lab in labels:
        if lab.hemi == 'lh':
            tmp_idx = np.nonzero(np.in1d(src[0]['vertno'], lab.vertices))[0]
        elif lab.hemi == 'rh':
            tmp_idx = src[0]['nuse'] + \
                np.nonzero(np.in1d(src[1]['vertno'], lab.vertices))[0]    
        parcels.append(tmp_idx)
        total_vertx[tmp_idx] = 0
        
    outliers = np.nonzero(total_vertx)[0]
    
    if outliers.size > 0:
        parcels.append(outliers) # For consistency with flame parcels
        name.append('Outliers')

    if type_index == 'matlab':
        parcels = map(lambda x:x+1, parcels)
        outliers = outliers + 1
    elif type_index == 'python':
        pass
    else:
        print('Type of indeces not understood')

    converted_labels = {'parcel' : parcels, 'name' : name, 
            'outliers_id' : outliers, 'outliers' : float(outliers.shape[0])}
    
    return converted_labels
