"""
Functions for plotting the computed MEG informed
cortical parcellations.

Authors: Sara Sommariva <sommariva@dima.unige.it>
"""
import numpy as np

from mne.viz import get_brain_class
from mne.utils import warn


# TODO: can I pass also the hemi in input??
def plot_flame_labels(indeces, labels, src, subject, subjects_dir,
                      surf='inflated', brain=None, color=None,
                      plot_region=True, plot_points=False, plot_borders=False):
    """Plot filled version of flame parcels

    Parameters
    ----------
    indeces : list of int
        List of regions to be plotted. Regions are identified by their position
        within flame parcellation. Indices start from 1
    labels : list of Labels
        Flame parcellation
    src : SourceSpaces
        Source space that has been parcellated
    subject : str
        Name of the subject
    subject_dir : str
        Path to SUBJECTS_DIR
    surf : str
        Freesurfer surface mesh name (ie 'white', 'inflated', etc.)
    brain : istance of Brain | None
        If None a new brain will be created
    color : tuple | None ?????
        Color to be used.
    plot_region : Boolean
        If True a colored, filled region will be plotted
    plot_points : Boolean
        If True the point within the region will be plotted
    plot_borders : Boolean
        If True the region borders will be plotted
    """

    labels_toplot = []
    for idx in indeces:
        aux_roi = [ir for ir in range(len(labels))
                   if 'roi%d-%s' % (idx, labels[ir].hemi) in labels[ir].name]
        if len(aux_roi) == 0:
            warn('No regions for index %d' % idx)
        labels_toplot += aux_roi
    nroi = len(labels_toplot)

    if color is not None:
        for ir in labels_toplot:
            aux = list(labels[ir].color)
            # aux[0:3] = color
            aux = color
            labels[ir].color = tuple(aux)

    if nroi == 0:
        raise ValueError('No region for the indeces in input')

    if brain is None:
        Brain = get_brain_class()
        brain = Brain(subject, hemi='both', surf=surf, background='white',
                      subjects_dir=subjects_dir, alpha=1)

    for ir in labels_toplot:
        # Plot regions
        if plot_region:
            brain.add_label(labels[ir].fill(src, name=None),
                            hemi=labels[ir].hemi)
            # Plot borders of regions
        if plot_borders:
            brain.add_label(labels[ir].fill(src, name=None),
                            hemi=labels[ir].hemi, borders=True)
        # Plot points
        if plot_points:
            brain.add_foci(labels[ir].vertices, coords_as_verts=True,
                           hemi=labels[ir].hemi, color=labels[ir].color[0:3],
                           scale_factor=0.3)

    return brain


def plot_flame_centroids(flame_data, src, subject, subjects_dir, brain=None,
                         surf='inflated', scale_factor=0.5, color='white'):
    if brain is None:
        Brain = get_brain_class()
        brain = Brain(subject, hemi='both', surf=surf, background='white',
                      subjects_dir=subjects_dir, alpha=1)

    nvert_lh = src[0]['nuse']
    idx_centr = flame_data['centroids_id']
    centr_lh = idx_centr[np.where(idx_centr < nvert_lh)]
    centr_rh = idx_centr[np.where(idx_centr >= nvert_lh)] - nvert_lh

    brain.add_foci(src[0]['vertno'][centr_lh], coords_as_verts=True,
                   hemi='lh', scale_factor=scale_factor, color=color)
    brain.add_foci(src[1]['vertno'][centr_rh], coords_as_verts=True,
                   hemi='rh', scale_factor=scale_factor, color=color)

    return brain
