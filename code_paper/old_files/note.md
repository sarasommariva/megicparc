Code analysis flame paper
=====
The results on flame pubblication can be replicated by running the following code.<br>
1. [analysis_run_flame.py](analysis_run_flame_meg.py) Compute flame parcellations and/or analyse their structural and functional properties via resolution matrices. <br>
2. [analysis_simulations_runs.py](analysis_simulations_runs.py) Simulate synthetic data. <br>
3. [analysis_simulations_analyse_results.py](analysis_simulations_analyse_results.py) Analyse results obtained in 2. <br>
4. [analysis_auditory_data.py](analysis_auditory_data.py) Analyse experimental data. <br>
5. [check_files.py](check_files.py) and [run_simulations_clusters.sh](run_simulations_clusters.sh) auxiliar files to run simulations in 2. on clusters.

analysis_run_flame.py
-----
How to use: <br>
First run with `do_compute_flame = True` to compute flame parcellations; then run with `do_compute_flame = False` to analyze
such parcellations. <br>
<br>
Data required:<br>
1. freesurfer subjects directories in `/m/nbe/work/sommars1/FLAME/subjects_flame` <br>
2. leadfield in `./data` <br>
3. flame parcellations saved in `./flame_parcellations/` (if not computed)

Folder(s) created (if not existing):<br>
1. `./figures` for storing pictures <br>
2. `./flame_parcellations` for saving the parcellation

analysis_simulations_run.py
-----
How to use: <br>
First run with `do_simulated_tc = True` to simulate the time-courses of the sources of interest; then run `run_simulations_clusters.sh` on clusters. Folder `.\job_out`, `.\job_err` and `.\test_simulation` have to be present.<br>
Run `check_files.py` to check that all files have been properly saved.
<br>
<br>
Data required:<br>
1. freesurfer subjects directories in `/m/nbe/work/sommars1/FLAME/subjects_flame` <br>
2. leadfield in `./data` <br>
3. flame parcellations in `./flame_parcellations/` <br>

Results are saved in: <br>
1. `./test_simulation/<subject>`

analysis_simulations_analyse_results.py
-----
Data required:<br>
1. Results of the simulation in `./test_simulation/<subject>` <br>



analysis_auditory_data.py
-----
Data required:<br>
1. freesurfer subjects directories in `/m/nbe/work/sommars1/FLAME/subjects_flame` <br>
2. leadfield in `./data` <br>
3. preprocessed MEG data in `./data` <br>
4. flame parcellations in `./flame_parcellations/` <br>
5. dipoles fitted with neuromag in `./data` <br>

General todo list
----
When working with python 3 neuroimaging environment I have some issues with mayavi.
