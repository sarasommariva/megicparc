# Check for missing file. It may happen e.g. that the job ran out of time.

import os.path as op
import pickle
import numpy as np

tot_job_run = 1000
subjects = ['k1_T1', 'k2_T1', 'k4_T1', 'k6_T1', 'k7_T1',
            'CC110045', 'CC110182', 'CC110174', 'CC110126', 'CC110056']
num_subs = len(subjects)

target = './'
folder_results = op.join(target, 'test_simulation')
folder_results_sub = op.join(folder_results, '{:s}')
string_results = op.join(folder_results_sub, 'result_grad_num_{:d}_{:s}.pkl')

print('Checking files in %s'%folder_results)

wrong_job = np.array([])

for aux_job_run in np.arange(tot_job_run)+1: # Job-run indeces as created by triton

    job_run = aux_job_run - 1
    i_run = job_run // num_subs
    subject_id = job_run%num_subs

    if subject_id == 0:
        print(i_run+1)
    subject = subjects[subject_id]
    path_res = string_results.format(subject, i_run+1, subject)

    try:
        pickle.load(open(path_res, 'rb'))
    except:
        print(path_res)
        wrong_job = np.append(wrong_job, aux_job_run)
        pass

print(wrong_job)
