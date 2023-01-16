# RL_Replay
Reinforcement learning experiment with different replay strategies and environment topologies

Supported replay models:
 - EVB (Expected Value of a Bellman backup - see Mattar and Daw, 2018)
 - gain_only (see Mattar and Daw, 2018)
 - need_only (see Mattar and Daw, 2018)
 - dyna (see Richard S Sutton, 1990)
 - no_replay
 - prioritized_sweeping (Cazé et al., 2018)

 To generate data for each model, run the run.py script, using a command prompt, or the analysis_e_greedy.ipynb notebook.
 Each model will be run on N simulations of 50 episodes on a single type of maze. Check the parameters.py file to modify N (10 by default) and other parameters. At episode 25, the reward function is modified, to observe the different models performances and adaptivity. It is possible to generate data for only a subset of models by removing the others of the 'models_dict' object in run.py.

 Different environment topologies are available to run the experiments, the default maze is a 6*9 grid called "mattar" and used in Mattar and Daw, (2018). Others are available like a 1x10 linear track for example. Check the run.py file for options. Comment your maze of interest and uncomment the others. The selected maze is plotted at the beginning of each call to run.py.

 The produced data can then be summary analyzed and plotted using only the analysis_e_greedy.ipynb notebook.
 Then again, only a subset of the generated data can be analyzed by removing models out of the "models" list in the notebook.
