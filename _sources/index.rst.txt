.. megicparc documentation master file, created by
   sphinx-quickstart on Thu Apr 27 11:14:31 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

megicparc: MEG leadfield-based cortical parcellations
=====================================================

Welcome to the documentation of megicparc, a Python3 library for dividing the cortical
surface in parcels whose activity can be faithfully represented by a single dipolar
source while minimizing inter-parcel crosstalk. The method is based on unsupervised 
clustering of the MEG lead fields, also accounting for distances between the cortically-constrained 
sources to promote spatially contiguous parcels.

Installation
============

To install the latest stable version of this package use ``pip``:

.. code::

    pip install megicparc

If you do not have admin privileges on the computer, use the ``--user`` flag
with ``pip``. 

To check if everything worked fine, you can run:

.. code::

    python -c 'import megicparc'

and it should not give any error messages.

Author of the code
==================
| Sara Sommariva <sommariva@dima.unige.it>,

Cite our work
=============

If you use this code in your project, please consider citing our work:

.. [1] S. Sommariva1, N. Puthanmadam Subramaniyam, and L. Parkkonen. An unsupervised clustering approach for MEG
lead-field-based cortical parcellation. Submitted

.. toctree::
    :hidden:

    api
    auto_examples/index
    auto_paper/index`
