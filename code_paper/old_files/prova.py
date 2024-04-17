import os.path as op
from mne import read_labels_from_annot, write_labels_to_annot
from mne.datasets import sample

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'sample'

parc = 'aparc'
#label_lh = read_labels_from_annot(subject=subject, parc=parc, hemi='lh',
#                                  subjects_dir=subjects_dir)
#label_rh = read_labels_from_annot(subject=subject, parc=parc, hemi='rh',
#                                  subjects_dir=subjects_dir)
#label = label_lh + label_rh

#write_labels_to_annot(label_rh, annot_fname='rh.prova.annot')
#write_labels_to_annot(label_lh, annot_fname='lh.prova.annot')

prova_lh = read_labels_from_annot(subject=subject,
                                  annot_fname='rh.aparc.annot')