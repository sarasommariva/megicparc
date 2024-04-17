#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:49:09 2019

@author: sommars1
"""

import numpy as np

def compute_inv_op_rank(L, C, lam, depth=None, method='MNE', rank=None):
    
    if depth == None:
        R = np.eye(L.shape[1])
    else:
        # Nota: funziona meglio quando uso L completa (Ns x 3 Nv)
        depth_weights = np.linalg.norm(L, ord=2, axis=0)**(-2.*depth)    
        R = np.diag(depth_weights)        
        
    if rank == None: 
        rank = L.shape[0]    
    
    # Step 1: Whitening
    _, S, V = np.linalg.svd(C)
    #ssv = np.where(S>toll*S[0])[0]
    ssv = np.arange(0, rank)
    print('Given rank = ' + str(rank) + ' - num eigenvalues = ' + str(ssv.shape[0]))
    W = np.dot(
         np.diag(1/np.sqrt(S[ssv])), V[ssv])
    Lw = np.dot(W, L)
    
    # Step 2: Rescale lambda
    RLt = R.dot(Lw.T)
    LRLt = Lw.dot(RLt)    
    
    lam = lam * np.trace(LRLt)/ssv.shape[0]
    
    # Step 3: Compute whitened inverse operator
    WMNEw = np.dot(RLt, np.linalg.inv(LRLt + lam * np.eye(ssv.shape[0])))   
    
    # Step 4: Compute the final inverse operator
    if method == 'MNE':
        inv_op = WMNEw.dot(W)
    elif method == 'dSPM':
        #weights = 1/np.sqrt(np.sum(WMNEw * WMNEw, axis=1))
        weights = 1/np.linalg.norm(WMNEw, ord=2, axis=1)
        WdSPMw = np.dot(np.diag(weights), WMNEw)
        inv_op = WdSPMw.dot(W)
    elif method == 'sLORETA': 
        weights = 1/np.sqrt(np.diag(WMNEw.dot(Lw.dot(R))))
        WsLORETAw = np.dot(np.diag(weights), WMNEw)
        inv_op = WsLORETAw.dot(W)
    
    return inv_op

def generate_surrogate(data, method):

    if method == 'shuffled':
        data_surr = np.random.permutation(data.transpose()).transpose()
        
    if method == 'random_phase':
        # Step 1. Compute FFT
        ft_x = np.fft.fft(data)
        Nd = data.shape[0]
        
        # Step 2. Randomize phase
        Nf = ft_x.shape[1]
        phase = np.zeros(ft_x.shape)
        if Nf/2 == np.ceil(Nf/2):
            Nf_2 = int(np.ceil(Nf/2))
            phase[:, 1:Nf_2] = 2*np.pi * np.random.rand(Nd, Nf_2-1) - np.pi
            phase[:, Nf_2+1:Nf] = -phase[:, Nf_2-1:0:-1]
            # Note: phase(0) = phase(-ft/2) = 0
        else:
            Nf_2 = int(np.ceil(Nf/2))
            phase[:, 1:Nf_2] = 2*np.pi * np.random.rand(Nd, Nf_2-1) - np.pi
            phase[:, Nf_2:Nf] = -phase[:, Nf_2-1:0:-1]
            # Note: phase(0) = 0

        # Step 3. Compute inverse FFT and take the real part (it should be already real)
        ft_xs = abs(ft_x)*np.exp(1j*phase)
        data_surr = np.real(np.fft.ifft(ft_xs))

    return data_surr

def collapse_RM(RM, labels, src, method):

    # Project anatomical ROI into source-space V and construct flip vectors
    nvert_lh = src[0]['nuse']       
    label_vertidx = list()
    label_flipvec = list()
    for label in labels: 
        if label.hemi == 'lh':
            tmp = np.in1d(src[0]['vertno'], label.vertices)
            # Label project in the source-space
            label_vertidx.append(np.nonzero(tmp)[0])
            # Flip vectors
            nn_label = src[0]['nn'][src[0]['vertno'][tmp]]
            nn_label.shape[0]
            _, _, Vnn = np.linalg.svd(nn_label)
            label_flipvec.append(np.sign(np.dot(nn_label, Vnn[0])))
        elif label.hemi == 'rh':
            tmp = np.in1d(src[1]['vertno'], label.vertices)
            # Label project in the source-space                
            label_vertidx.append(nvert_lh + np.nonzero(tmp)[0])
            # Flip vectors            
            nn_label = src[1]['nn'][src[1]['vertno'][tmp]]
            _, _, Vnn = np.linalg.svd(nn_label)
            label_flipvec.append(np.sign(np.dot(nn_label, Vnn[0])))
        else:
            print('BiHemi Label')
            
    # Project RM
    n_rois = len(labels)
    ROI_RM = np.zeros([n_rois, RM.shape[1]])
    # Method 1: Mean over the points of the ROIs
    if method == 'mean':
        for ir, idx in enumerate(label_vertidx):
            ROI_RM[ir] = np.mean(RM[idx], axis=0) 
    # Method 2: Flipped mean
    elif method == 'mean_flip':
        # Step 2a. Flip 
        for ir, (idx, flip) in enumerate(zip(label_vertidx, label_flipvec)):
            ROI_RM[ir] = np.average(RM[idx].T*flip, axis=1)
    else:
        print('Method still not understood')
    

    return ROI_RM

def window_data(data, Twin, Overlap, ft):
    # INPUT: 
    # 1) data: ndarray [NsxNt]
    #    Original data
    # 2) Twin : float
    #    Lenght of the data [sec]
    # 3) Overlap : float
    #    Portion of overlapping window
    # 4) ft : float
    #    Sampling frequency [Hz]

    # TODO: prendere qualche parte intera:
    Ns, Nt = data.shape    
    Nwin = int(Twin * ft)
    D = int(Overlap*Nwin)
    Nep = int(np.floor((Nt-Nwin)/D) + 1)

    epoched_data = np.zeros([Nep, Ns, Nwin])
    
    for iep in range(Nep):
        epoched_data[iep, :, :] = data[:, D*iep:D*iep+Nwin]
    
    return epoched_data

def compute_auc(estimated_values, true_roi):
    """
    Compute the value of the area under the receiver operating characteristic
    curve (AUC).
    
    Parameters:
    -----------
    estimated_values: ndarray of float [Nroi x Nroi]
        Estimated connectivity values for a parcellation
    true_roi: array of int [2, ]
        Indices of the regions containing the interacting sources
    
    Returns:
    --------
    auc : float
       AUC value
    """

    from sklearn.metrics import roc_auc_score
    
    true_roi = np.sort(true_roi)[::-1]
    n_roi = estimated_values.shape[0]

    significant_true = np.zeros([n_roi, n_roi])
    significant_true[true_roi[0], true_roi[1]] = 1

    _mask = np.tril_indices(n_roi, k=-1)

    auc = roc_auc_score(significant_true[_mask], estimated_values[_mask])

    return auc

def read_dipole_locations(fname):
    
    import re
    from mne.utils import logger, warn
    
    """Read a dipole text file."""
    # Function adapted from mne-python '_read_dipole_text'.
    # https://github.com/mne-tools/mne-python/blob/maint/0.17/mne/dipole.py#L491-L518
    # Only dipole locations are read. The mne-python function cannot be used
    # directly beacuse I did not store the dipole moments
    
    # Figure out the special fields
    need_header = True
    def_line = name = None
    # There is a bug in older np.loadtxt regarding skipping fields,
    # so just read the data ourselves (need to get name and header anyway)
    data = list()
    with open(fname, 'r') as fid:
        for line in fid:
            if not (line.startswith('%') or line.startswith('#')):
                need_header = False
                data.append(line.strip().split())
            else:
                if need_header:
                    def_line = line
                if line.startswith('##') or line.startswith('%%'):
                    m = re.search('Name "(.*) dipoles"', line)
                    if m:
                        name = m.group(1)
        del line
        
    data = np.atleast_2d(np.array(data, float))
    if def_line is None:
        raise IOError('Dipole text file is missing field definition '
                      'comment, cannot parse %s' % (fname,))
    # actually parse the fields
    def_line = def_line.lstrip('%').lstrip('#').strip()
    # MNE writes it out differently than Elekta, let's standardize them...
    fields = re.sub(r'([X|Y|Z] )\(mm\)',  # "X (mm)", etc.
                    lambda match: match.group(1).strip() + '/mm', def_line)
    fields = re.sub(r'\((.*?)\)',  # "Q(nAm)", etc.
                    lambda match: '/' + match.group(1), fields)
    fields = re.sub('(begin|end) ',  # "begin" and "end" with no units
                    lambda match: match.group(1) + '/ms', fields)
    fields = fields.lower().split()
    required_fields = ('begin/ms',
                       'x/mm', 'y/mm', 'z/mm')
#                       'q/nam', 'qx/nam', 'qy/nam', 'qz/nam',
#                       'g/%')
    optional_fields = ('khi^2', 'free',  # standard ones
                       # now the confidence fields (up to 5!)
                       'vol/mm^3', 'depth/mm', 'long/mm', 'trans/mm',
                       'qlong/nam', 'qtrans/nam', 
                       'q/nam', 'qx/nam', 'qy/nam', 'qz/nam',
                       'g/%')
    conf_scales = [1e-9, 1e-3, 1e-3, 1e-3, 1e-9, 1e-9]
    missing_fields = sorted(set(required_fields) - set(fields))
    if len(missing_fields) > 0:
        raise RuntimeError('Could not find necessary fields in header: %s'
                           % (missing_fields,))
    handled_fields = set(required_fields) | set(optional_fields)
    assert len(handled_fields) == len(required_fields) + len(optional_fields)
    ignored_fields = sorted(set(fields) -
                            set(handled_fields) -
                            set(['end/ms']))
    if len(ignored_fields) > 0:
        warn('Ignoring extra fields in dipole file: %s' % (ignored_fields,))
    if len(fields) != data.shape[1]:
        raise IOError('More data fields (%s) found than data columns (%s): %s'
                      % (len(fields), data.shape[1], fields))

    logger.info("%d dipole(s) found" % len(data))

    if 'end/ms' in fields:
        if np.diff(data[:, [fields.index('begin/ms'),
                            fields.index('end/ms')]], 1, -1).any():
            warn('begin and end fields differed, but only begin will be used '
                 'to store time values')

    # Find the correct column in our data array, then scale to proper units
    idx = [fields.index(field) for field in required_fields]
#    assert len(idx) >= 9
#    times = data[:, idx[0]] / 1000.
    pos = 1e-3 * data[:, idx[1:4]]  # put data in meters
    
    return pos
