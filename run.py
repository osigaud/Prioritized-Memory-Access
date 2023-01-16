from parameters import setParams
import matplotlib.pyplot as plt
from main import Replay_Sim
from copy import copy
import numpy as np
import glob
import os

from myMaze import CustomMaze

models_dict = {
     'prioritized_sweeping': {'n_plan': 20, 'set_gain_to_one': True, 'set_need_to_one': True},
     'dyna': {'n_plan': 20, 'set_gain_to_one': True, 'set_need_to_one': True},
     'no_replay': {'n_plan': 0, 'set_gain_to_one': True, 'set_need_to_one': True},
     'need_only': {'n_plan': 20, 'set_gain_to_one': True, 'set_need_to_one': False},
     'gain_only': {'n_plan': 20, 'set_gain_to_one': False, 'set_need_to_one': True},
     'EVB': {'n_plan': 20, 'set_gain_to_one': False, 'set_need_to_one': False}
}

# maze_dict = {'large_open': {
#     'size': (20, 30),
#     'walls': [[slice(1, 4), 2], [slice(0, 2), 4], [4, 5],
#               [3, 4], [3, slice(7, 9)], [4, 1], [slice(9, 15), 12],
#               [slice(17, 20), 14], [12, 13], [8, 25], [13, slice(17, 19)],
#               [4, 11], [slice(11, 14), 12], [slice(5, 7), 24], [14, 25],
#               [13, 24], [13, slice(27, 29)], [14, 21], [slice(6, 20), 0],
#               [19, slice(0, 30)], [slice(0, 20), 29], [0, slice(4, 30)],
#               [slice(14, 20), 1], [18, slice(1, 6)], [17, slice(1, 5)],
#               [16, slice(1, 4)], [slice(15, 19), 2], [4, 1], [18, 28],
#               [1, 28]
#               ],
#     'start_state': np.array([[10, 1]]),
#     'goal_state.s_1': np.array([[2, 28]]),
#     'goal_state.s_2': np.array([[0, 2]]),
#     'reward_magnitude.s': np.array([[1]]),
#     'reward_std.s': np.array([[0.1]]),
#     'reward_prob.s': np.array([[1]])
# }, 'mattar': {
#     'size': (6, 9),
#     'walls': [[slice(1, 4), 2], [slice(0, 3), 7], [4, 5]],
#     'start_state': np.array([[2, 0]]),
#     'goal_state.s_1': np.array([[0, 8]]),
#     'goal_state.s_2': np.array([[5, 0]]),
#     'reward_magnitude.s': np.array([[1]]),
#     'reward_std.s': np.array([[0.1]]),
#     'reward_prob.s': np.array([[1]])
# }, 'mattar_open': {
#     'size': (6, 9),
#     'walls': [[5, slice(2, 9)], [0, slice(0, 5)],
#               [slice(3, 5), 8]],
#     'start_state': np.array([[2, 0]]),
#     'goal_state.s_1': np.array([[0, 8]]),
#     'goal_state.s_2': np.array([[5, 0]]),
#     'reward_magnitude.s': np.array([[1]]),
#     'reward_std.s': np.array([[0.1]]),
#     'reward_prob.s': np.array([[1]])
# }, 'mattar_constrained': {
#     'size': (6, 9),
#     'walls': [[slice(1, 4), 2], [slice(0, 5), 7], [slice(2, 5), 5],
#               [1, 3], [4, 4], [3, 1]],
#     'start_state': np.array([[2, 0]]),
#     'goal_state.s_1': np.array([[0, 8]]),
#     'goal_state.s_2': np.array([[5, 0]]),
#     'reward_magnitude.s': np.array([[1]]),
#     'reward_std.s': np.array([[0.1]]),
#     'reward_prob.s': np.array([[1]])
# }}


# maze_dict = {'t_maze': {
    # 'size': (1, 10),
    # 'walls': [],
    # 'start_state': np.array([[0, 0]]),
    # 'goal_state.s_1': np.array([[0, 9]]),
    # 'goal_state.s_2': np.array([[0, 9]]),
    # 'reward_magnitude.s': np.array([[1]]),
    # 'reward_std.s': np.array([[0.1]]),
    # 'reward_prob.s': np.array([[1]])
 # },}


# maze_dict = {'spec_maze': {
    # 'size': (5, 9),
    # 'walls': [(1,i) for i in range(1,5)]  + [(3,i) for i in range(4,9)] + [(2,1)] + [(3,5)] + [(2,4)] + [(3,4)],
    # 'start_state': np.array([[0, 8]]),
    # 'goal_state.s_1': np.array([[0, 4]]),
    # 'goal_state.s_2': np.array([[4, 8 ]]),
    # 'reward_magnitude.s': np.array([[1]]),
    # 'reward_std.s': np.array([[0.1]]),
    # 'reward_prob.s': np.array([[1]])
 # },}


maze_dict = {'mattar': {
    'size': (6, 9),
    'walls': [[slice(1, 4), 2], [slice(0, 3), 7], [4, 5]],
    'start_state': np.array([[2, 0]]),
    'goal_state.s_1': np.array([[0, 8]]),
    'goal_state.s_2': np.array([[5, 0]]),
    'reward_magnitude.s': np.array([[1]]),
    'reward_std.s': np.array([[0.1]]),
    'reward_prob.s': np.array([[1]])
 },}

# custom = CustomMaze()
# maze_dict = {"custom":custom.generateMaze(10)}

params = setParams()

# iterate over each maze (different environment topologies, single/double reward)

np.random.seed(31415)
for maze in maze_dict:
    print('\nStarting simulations for the {} maze\n'.format(maze))
    # create maze
    maze_size = maze_dict[maze]['size']
    params.maze = np.zeros(maze_size)
    # add walls
    walls = maze_dict[maze]['walls']

    for wall in walls:
        params.maze[wall[0], wall[1]] = 1

    tmp2 = copy(params.maze)
    tmp2[maze_dict[maze]["start_state"][0][0], maze_dict[maze]["start_state"][0][1]] = 2
    tmp2[maze_dict[maze]["goal_state.s_1"][0][0], maze_dict[maze]["goal_state.s_1"][0][1]] = 0.5
    print("Using "+maze+" maze. Starting position in yellow, walls in light blue, pre transition reward in dark blue")
    plt.imshow(tmp2, interpolation='none')
    plt.show()

    tmp2 = copy(params.maze)
    tmp2[maze_dict[maze]["start_state"][0][0], maze_dict[maze]["start_state"][0][1]] = 2
    tmp2[maze_dict[maze]["goal_state.s_2"][0][0], maze_dict[maze]["goal_state.s_2"][0][1]] = 0.5
    print("After reward function change, reward in dark blue")
    plt.imshow(tmp2, interpolation='none')
    plt.show()

    print()

    # starting state of the agent (in matrix notation)
    params.s_start = maze_dict[maze]['start_state']
    # set random starting point to True or False
    params.s_start_rand = True
    # choose policy ('e_greedy' or 'softmax')
    params.actPolicy = 'e_greedy'
    # set probability of a random action(epsilon - greedy)
    params.epsilon = 0.05
    # goal state(s) (in matrix notation)
    params.s_end = maze_dict[maze]['goal_state.s_1']
    params.s_end_change = maze_dict[maze]['goal_state.s_2']
    # reward magnitude (rows: locations; columns: values)
    params.rewMag = maze_dict[maze]['reward_magnitude.s']
    # reward Gaussian noise (rows: locations; columns: values)
    params.rewSTD = maze_dict[maze]['reward_std.s']
    # probability of receiving each reward (columns: values)
    params.rewProb = maze_dict[maze]['reward_prob.s']

    # iterate over each model (i.e. replay strategies)
    for model in models_dict:
        # number of steps to do in planning (set to zero if no planning or
        # to Inf to plan for as long as it is worth it)
        params.nPlan = models_dict[model]['n_plan']
        params.setAllGainToOne = models_dict[model]['set_gain_to_one']
        params.setAllNeedToOne = models_dict[model]['set_need_to_one']

        # loop over each model and maze for multiple simulation
        file_list = [file for file in glob.glob(os.path.join('checkpoints', maze, model + '*'))]
        # loop over each model and maze for multiple simulation (that are not already saved as a chekpoint)
        for k in range(len(file_list), params.N_SIMULATIONS):
            print("Simulation number: ", k)
            np.random.seed()
            ReplayModel = Replay_Sim(params, model, maze, sim_i=k)
            # pre-explore the environment/maze
            ReplayModel.pre_explore_env()
            # build a transition function/matrix based on the pre-exploration
            ReplayModel.build_transition_mat()
            # explore the environment/maze (i.e. start episode)
            ReplayModel.explore_env()
            #print()
            #print(ReplayModel.performance_df)
            # save simulation
            ReplayModel.save()
            del ReplayModel
            progress = "\nDone with {} simulation #{} out of #{}".format(model, str(k + 1), str(params.N_SIMULATIONS))
            print(progress)
