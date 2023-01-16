import numpy as np
from numpy import matlib
import time
import pickle
import scipy
import matplotlib.pyplot as plt
import os
import pandas as pd


class Replay_Sim:
    """
    Replay_Sim: Replay simulation on a grid-world.
    params = Structure with all of the simulation parameters
    This code was written by Roy Tal (roy.dew@gmail.com)
    Replication and extension of MATLAB code by Marcelo G Mattar (mmattar@princeton.edu), Jan 2017
    Mattar's code is available here: https://github.com/marcelomattar/PrioritizedReplay
    Mattar's paper is available here: https://www.nature.com/articles/s41593-018-0232-z
    """

    def __init__(self, params, replay_strategy, maze, sim_i=''):
        """
        params: simulation parameters
        replay_strategy = 'prioritized_sweeping', 'EVB', 'no_replay', 'dyna', 'gain_only' or 'need_only'
        sim_i: index in simulation loop
        """
        acceptable_replay_strategies = ['prioritized_sweeping', 'EVB', 'no_replay', 'dyna', 'gain_only', 'need_only']
        assert replay_strategy in acceptable_replay_strategies, "'{}' is not an acceptable replay strategy, " \
                                                                "please select one of the following strategies:\n" \
                                                                "'prioritized_sweeping', 'EVB', 'no_replay', 'dyna', " \
                                                                "'gain_only' or 'need_only'".format(replay_strategy)
        # INITIALIZE VARIABLES
        self.params = params
        self.params_dict = {}
        for key in self.params.__dict__:
            if not key.startswith('__'):
                self.params_dict[key] = self.params.__dict__[key]
        self.replay_strategy = replay_strategy
        self.sim_i = str(sim_i)
        self.maze = maze
        # make directory for this maze
        if maze not in os.listdir('checkpoints/'):
            os.mkdir(os.path.join('checkpoints', maze))

        side_ii, side_jj = params.maze.shape  # get the initial maze dimensions
        self.side_ii = side_ii
        self.side_jj = side_jj
        self.first_goal = self.params.s_end
        self.second_goal = self.params.s_end_change
        self.this_goal = np.copy(self.params.s_end)
        self.last_tsi_at_goal_1 = np.nan
        self.first_tsi_at_goal_2 = np.nan
        self.this_starting_state_i = np.nan

        self.n_states = side_ii * side_jj  # maximal number of states
        self.n_actions = 4  # number of actions available at each state
        self.Q = np.zeros((self.n_states, self.n_actions))  # state-action value function
        self.T = np.zeros((self.n_states, self.n_states))  # state-state transition probability
        self.elig_trace = np.zeros((self.n_states, self.n_actions))  # eligibility matrix
        self.PQueue = []

        self.exp_arr_full = np.empty((0, 4))  # array to store all individual experiences (with pre-exploration)
        self.exp_arr = np.empty((0, 4))  # array to store individual experiences from exploration only

        self.exp_last_stp1 = np.full((self.n_states, self.n_actions), np.nan)  # <- next state
        self.exp_last_rew = np.full((self.n_states, self.n_actions), np.nan)  # <- immediate reward
        # self.steps_per_episode = np.full(self.params.MAX_N_EPISODES, np.nan)
        empty_array = np.full(self.params.MAX_N_EPISODES, np.nan)
        data = {'steps_per_episode': empty_array,
                'full_time_per_episode': empty_array,
                'n_plan_phases_g1': empty_array,
                'n_plan_phases_g2': empty_array,
                'n_plan_phases_g1_to_s': empty_array,
                'n_plan_phases_g2_to_s': empty_array,
                'n_plan_steps_g1': empty_array,
                'n_plan_steps_g2': empty_array,
                'n_plan_steps_g1_to_s': empty_array,
                'n_plan_steps_g2_to_s': empty_array,
                'plan_times_g1': empty_array,
                'plan_times_g2': empty_array,
                'plan_times_g1_to_s': empty_array,
                'plan_times_g2_to_s': empty_array,
                'gain_times_g1': empty_array,
                'gain_times_g2': empty_array,
                'gain_times_g1_to_s': empty_array,
                'gain_times_g2_to_s': empty_array,
                'need_times_g1': empty_array,
                'need_times_g2': empty_array,
                'need_times_g1_to_s': empty_array,
                'need_times_g2_to_s': empty_array,
                'EVB_times_g1': empty_array,
                'EVB_times_g2': empty_array,
                'EVB_times_g1_to_s': empty_array,
                'EVB_times_g2_to_s': empty_array,
                'PS_prep_times_g1': empty_array,
                'PS_prep_times_g2': empty_array,
                'PS_prep_times_g1_to_s': empty_array,
                'PS_prep_times_g2_to_s': empty_array
                }
        self.performance_df = pd.DataFrame(data)

        self.num_episodes = 0  # <- keep track of how many times we reach the end of our maze
        self.num_episodes_arr = np.empty((self.params.MAX_N_STEPS, 1))

        self.replay = {'state': np.full((self.params.MAX_N_STEPS, self.params.nPlan), np.nan),
                       'action': np.full((self.params.MAX_N_STEPS, self.params.nPlan), np.nan),
                       'gain': np.full((self.params.MAX_N_STEPS, self.params.nPlan), np.nan),
                       'need': np.full((self.params.MAX_N_STEPS, self.params.nPlan), np.nan),
                       'EVM': np.full((self.params.MAX_N_STEPS, self.params.nPlan), np.nan),
                       'TD': np.full((self.params.MAX_N_STEPS, self.params.nPlan), np.nan)}

    def save(self):
        with open('checkpoints/' + self.maze + '/' + self.replay_strategy + self.sim_i + '.pkl', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open('checkpoints/' + self.maze + '/' + self.replay_strategy + '.pkl', 'rb') as f:
            return pickle.load(f)

    def plot_agent(self, st):
        if self.params.plot_agent:
            plt.figure(1)
            maze_im = np.copy(self.params.maze)
            maze_im[st[0], st[1]] = 0.5
            plt.imshow(maze_im)
            plt.text(self.this_goal[0][1], self.this_goal[0][0],
                     'G', color='white', size=20, ha='center', va='center')
            plt.pause(0.01)

    def plot_Q_map(self):
        if self.params.plot_Q:
            plt.figure(2)
            plt.clf()
            plt.subplot(332)
            plt.imshow(self.Q[:, 0].reshape((self.side_ii, self.side_jj)))
            plt.title('Q Values (Up)')
            plt.subplot(338)
            plt.imshow(self.Q[:, 1].reshape((self.side_ii, self.side_jj)))
            plt.title('Q Values (Down)')
            plt.subplot(336)
            plt.imshow(self.Q[:, 2].reshape((self.side_ii, self.side_jj)))
            plt.title('Q Values (Right)')
            plt.subplot(334)
            plt.imshow(self.Q[:, 3].reshape((self.side_ii, self.side_jj)))
            plt.title('Q Values (Left)')
            plt.pause(0.01)

    def sample_transition(self, st, at):

        """
        Input state and action, return next state and reward
        """

        # convert to row/column notation:
        ii = st[0]
        jj = st[1]

        # MOVE THE AGENT TO THE NEXT POSITION

        # incorporate any actions and fix position if agent ends up outside of the grid:
        if at == 0:
            # action = UP
            stp1 = [ii - 1, jj]
        elif at == 1:
            # action = DOWN
            stp1 = [ii + 1, jj]
        elif at == 2:
            # action = RIGHT
            stp1 = [ii, jj + 1]
        elif at == 3:
            # action = LEFT
            stp1 = [ii, jj - 1]
        else:
            raise ValueError('Unknown value for action = {}'.format(at))

        # adjust the agent's position if it has fallen outside of the grid:
        if stp1[0] < 0:
            stp1[0] = 0
        if stp1[0] > self.side_ii - 1:
            stp1[0] = self.side_ii - 1
        if stp1[1] < 0:
            stp1[1] = 0
        if stp1[1] > self.side_jj - 1:
            stp1[1] = self.side_jj - 1

        if self.params.maze[stp1[0], stp1[1]] == 1:
            stp1 = st

        # convert to an index:
        stp1i = np.ravel_multi_index(stp1, [self.side_ii, self.side_jj])

        # COLLECT REWARD

        if isinstance(stp1, np.ndarray):  # make sure that stp1 is a list
            stp1 = stp1.tolist()

        # if this transition leads to a wall, no transition takes place

        if stp1 in self.this_goal.tolist():  # if the successor state is a goal state
            # if there's a different reward for each goal state:
            if self.params.rewMag.shape[0] == self.this_goal.shape[0]:
                # select reward magnitudes for the goal:
                this_rew = self.params.rewMag[np.all(self.this_goal == stp1, axis=1)]
            else:
                this_rew = self.params.rewMag[0]
            # if there's a different std for each goal state:
            if self.params.rewSTD.shape[0] == self.this_goal.shape[0]:
                # select std for the goal:
                this_std = self.params.rewSTD[np.all(self.this_goal == stp1, axis=1)]
            else:
                this_std = self.params.rewSTD[0]
            # Draw a sample from the reward magnitude list

            # if len(this_rew.shape) > 1 and len(params.rewProb.shape) > 1:

            if this_rew.shape[1] == self.params.rewProb.shape[1]:
                this_prob = self.params.rewProb / np.sum(self.params.rewProb)
                rew_idx = self.probs_to_index(this_prob)
                this_rew = this_rew[:, rew_idx][0]
            else:  # If the probabilities were not correctly specified, use the first column only
                rew_idx = 0
                this_rew = this_rew[:, rew_idx][0]

            # Draw a sample from the reward std list
            if this_std.shape[1] == self.params.rewProb.shape[1]:
                this_std = this_std[:, rew_idx][0]
            else:
                this_std = this_std[:, 0][0]

            # Compute the final reward
            rew = this_rew + np.random.normal() * this_std
            if self.params.rewOnlyPositive:
                rew = max(0, rew)  # Make sure reward + noise is positive (negative rewards are weird)
        else:
            rew = 0
        return [rew, stp1, stp1i]

    def probs_to_index(self, probs):
        rand = np.random.rand()
        cum_sum = np.cumsum(np.insert(probs, 0, 0))
        return np.argwhere(rand > cum_sum)[-1][0]

    def get_valid_states(self):
        # can start at any non-wall state
        valid_states = np.array(
            [np.ravel_multi_index(st, [self.side_ii, self.side_jj])
             for st in np.argwhere(self.params.maze == 0)])
        # but remove the goal states from the list
        goal_states = np.ravel_multi_index(
            [self.this_goal[:, 0], self.this_goal[:, 1]], [self.side_ii, self.side_jj])
        mask = np.invert(np.isin(valid_states, goal_states))
        return valid_states[mask]

    def get_act_probs(self, Q_mean):  # ADD?!? Q_var: |A|x1 array with the variance of the Q-value for each action
        """
        Calculate the probability of executing each action.
        Q_mean: |A|x1 array with the mean Q-value for each action
        """

        # INITIALIZE VARIABLES
        probs = np.full(Q_mean.shape[0], np.nan)
        probs = np.expand_dims(probs, axis=0)
        Q_mean = np.expand_dims(Q_mean, axis=0)

        # IMPLEMENT ACTION SELECTION STRATEGY
        if self.params.actPolicy == 'e_greedy':
            # notice that this loops through states, but often times this function receives only one state
            for s in range(Q_mean.shape[0]):
                Q_opt = np.argwhere(Q_mean[s] == np.amax(Q_mean[s]))  # Find indices of actions with maximum value
                if np.all(Q_mean[s] == np.amax(Q_mean[s])):
                    probs[s, Q_opt] = 1 / len(Q_opt.flatten())
                else:
                    probs[s, Q_opt] = (1 - self.params.epsilon) / len(Q_opt.flatten())
                    probs[np.isnan(probs)] = 0
                    # with probability epsilon, pick a random action:
                    probs += self.params.epsilon / len(Q_mean[s].flatten())

        elif self.params.actPolicy == 'softmax':
            for s in range(Q_mean.shape[0]):
                probs[s] = np.divide(np.exp(self.params.softmaxInvT * Q_mean[s]),
                                     np.sum(np.exp(self.params.softmaxInvT * Q_mean[s])))
        else:
            err_msg = "'{}' is an unrecognized strategy; use either 'e_greedy' or 'softmax')"
            raise ValueError(err_msg.format(self.params.actPolicy))
        rows_sum_to_1 = abs(np.sum(probs, axis=1) - 1) < 1e-3  # boolean array checking if each row sums to 1
        if np.all(rows_sum_to_1):  # return probabilities if all rows sum to 1
            return probs
        else:  # otherwise, raise an error
            # get string detailing the indices of rows that do not sum to 1 (erroneous rows)
            not_sum_1_indices = ''.join(c for c in str(np.argwhere(~rows_sum_to_1)) if c not in '[]\n').replace(' ',
                                                                                                                ', ')
            raise Exception('Probabilities of the following row(s) do not sum to 1: {}'.format(not_sum_1_indices))

    def get_starting_state(self):
        if not self.params.s_start_rand:  # if starting locations are not randomized
            st = self.params.s_start[0, :]
            sti = np.ravel_multi_index(st, [self.side_ii, self.side_jj])
        else:  # if starting locations are randomized
            st = [None, None]
            valid_states = self.get_valid_states()
            sti = valid_states[np.random.randint(len(valid_states))]
            st[0], st[1] = np.unravel_index(sti, [self.side_ii, self.side_jj])  # Convert state index to (i,j)
        self.this_starting_state_i = sti
        return st, sti

    def sample_action(self, sti):
        probs = self.get_act_probs(self.Q[sti])
        at = self.probs_to_index(probs)  # select an action
        return at

    def update_transition_mat(self, sti, stp1i):
        target_vec = np.zeros((1, self.n_states))
        target_vec[0][stp1i] = 1
        # shift corresponding row of T towards target_vec
        self.T[sti] += self.params.TLearnRate * (target_vec[0] - self.T[sti])

    def Q_learning(self, sti, at, rew, stp1i):  # ADD PLAN Q_LEARNING UPDATED TO THIS FUNCTION
        if self.params.onVSoffPolicy == 'on-policy':
            # Expected SARSA (learns Qpi)
            stp1_value = np.sum(np.multiply(self.Q[stp1i],
                                            self.get_act_probs(self.Q[stp1i])))
        else:
            stp1_value = np.max(self.Q[stp1i])  # Q-learning (learns Q*)
        delta = rew + self.params.gamma * stp1_value - self.Q[sti, at]  # prediction error (Q-learning)
        if self.replay_strategy == 'prioritized_sweeping':
            self.update_p_queue(delta, sti, at, rew, stp1i)

        # Update eligibility trace (eTr) using replacing traces, http://www.incompleteideas.net/book/ebook/node80.html)
        self.elig_trace[sti, at] = 1
        self.elig_trace[sti, np.argwhere(np.arange(self.n_actions) != at)] = 0
        self.Q += (self.params.alpha * self.elig_trace) * delta  # TD-learning
        self.elig_trace = self.elig_trace * self.params.lamb * self.params.gamma  # decay eligibility trace

    def update_p_queue(self, delta, sti, at, rew, stp1i):
        in_buffer = False
        abs_delta = np.abs(delta)
        if abs_delta > self.params.PS_thresh:
            # check whether experience/transition is already in the memory buffer (self.PQueue)
            for i, exp in enumerate(self.PQueue):
                if exp[1]['sti'] == sti and exp[1]['at'] == at:
                    in_buffer = True
                    # only add new experience if it is in the buffer and has greater abs(TD) than previous experience
                    if abs_delta > exp[0]:
                        self.PQueue[i] = (abs_delta, {"sti": sti, "at": at, "rew": rew, "stp1i": stp1i})
            if not in_buffer:
                self.PQueue.append((abs_delta, {"sti": sti, "at": at, "rew": rew, "stp1i": stp1i}))

    def get_gain(self, plan_exp):
        gain = []
        sa_gain = np.empty(self.Q.shape)
        sa_gain.fill(np.nan)
        for i in range(len(plan_exp)):
            this_exp = plan_exp[i]
            if len(this_exp.shape) == 1:
                this_exp = np.expand_dims(this_exp, axis=0)
            gain.append(np.repeat(np.nan, this_exp.shape[0]))

            for j in range(this_exp.shape[0]):
                Q_mean = np.copy(self.Q[int(this_exp[j, 0])])
                Qpre = Q_mean  # NOT USING THIS??
                # Policy BEFORE backup
                pA_pre = self.get_act_probs(Q_mean)

                # Value of state stp1
                stp1i = int(this_exp[-1, 3])
                if self.params.onVSoffPolicy == 'on-policy':
                    stp1_value = np.sum(np.multiply(self.Q[stp1i], self.get_act_probs(self.Q[stp1i])))
                else:
                    stp1_value = np.max(self.Q[stp1i])

                act_taken = int(this_exp[j, 1])
                steps_to_end = this_exp.shape[0] - (j + 1)
                rew = np.dot(np.power(self.params.gamma, np.arange(0, steps_to_end + 1)), this_exp[j:, 2])
                Q_target = rew + np.power(self.params.gamma, steps_to_end + 1) * stp1_value
                if self.params.copyQinPlanBkps:
                    Q_mean[act_taken] = Q_target
                else:
                    Q_mean[act_taken] += self.params.alpha * (Q_target - Q_mean[act_taken])

                # policy AFTER backup
                pA_post = self.get_act_probs(Q_mean)

                # calculate gain
                EV_pre = np.sum(np.multiply(pA_pre, Q_mean))
                EV_post = np.sum(np.multiply(pA_post, Q_mean))
                gain[i][j] = EV_post - EV_pre
                Qpost = Q_mean  # NOT USING THIS??

                # Save on gain[s, a]
                sti = int(this_exp[j, 0])
                if np.isnan(sa_gain[sti, act_taken]):
                    sa_gain[sti, act_taken] = gain[i][j]
                else:
                    sa_gain[sti, act_taken] = max(sa_gain[sti, act_taken], gain[i][j])
        return gain, sa_gain

    def get_need(self, sti, plan_exp):

        need = []
        if self.params.onlineVSoffline == 'online':  # Calculate Successor Representation
            SR = np.linalg.inv(np.eye(self.T.shape[0]) - self.params.gamma * self.T)
            SRi = SR[sti]  # Calculate the Successor Representation for the current state
            SR_or_SD = SRi
        elif self.params.onlineVSoffline == 'offline':  # Calculate eigenvectors and eigenvalues
            eig_vals, r_eig_vecs = np.linalg.eig(self.T)
            l_eig_vecs = scipy.linalg.eig(self.T, left=True, right=False)[1]
            assert abs(eig_vals[0]) - 1 > 1e-10, 'Precision error'
            SD = abs(l_eig_vecs[:, 0])  # Stationary distribution of the MDP
            SR_or_SD = SD

        # Calculate need-term for each experience in nStepExps
        for i in range(len(plan_exp)):
            this_exp = plan_exp[i]
            if len(this_exp.shape) == 1:
                this_exp = np.expand_dims(this_exp, axis=0)
            need.append(np.repeat(np.nan, this_exp.shape[0]))
            for j in range(this_exp.shape[0]):
                need[i][j] = SR_or_SD[int(this_exp[j, 0])]
        return need, SR_or_SD

    def check_if_goal_step(self, curr_or_prev, goal_type):
        if goal_type == 'this_goal':
            goal = self.this_goal
        elif goal_type == 'first_goal':
            goal = self.first_goal
        elif goal_type == 'second_goal':
            goal = self.second_goal
        if curr_or_prev == 'curr':
            is_goal_step = self.exp_arr_full[-1, 3] in np.ravel_multi_index(
                [goal[:, 0], goal[:, 1]], [self.side_ii, self.side_jj])
        elif curr_or_prev == 'prev':
            is_goal_step = [self.exp_arr_full[-2, 0], self.exp_arr_full[-2, 3]] == [np.ravel_multi_index(
                [goal[:, 0], goal[:, 1]], [self.side_ii, self.side_jj]), self.this_starting_state_i]
            # assert not self.params.s_start_rand, "Needs to be non-random starting state to assess this transition"
            # is_goal_step = np.all(np.isin([self.exp_arr_full[-2, 3], self.exp_arr_full[-2, 0]], [np.ravel_multi_index(
            #     [goal[:, 0], goal[:, 1]], [self.side_ii, self.side_jj]), np.ravel_multi_index(
            #     [self.params.s_start[:, 0], self.params.s_start[:, 1]], [self.side_ii, self.side_jj])]))
        return is_goal_step

    def plan(self, rew, sti):

        # PLANNING PREP
        p = 0  # Initialize planning step counter
        if self.params.planOnlyAtGorS:  # Only do replay if either current or last trial was a goal state
            # Current step is a move towards a goal
            curr_step_is_goal = self.check_if_goal_step('curr', 'this_goal')
            # curr_step_is_goal = self.exp_arr_full[-1, 3] in np.ravel_multi_index(
            #     [self.this_goal[:, 0], self.this_goal[:, 1]], [self.side_ii, self.side_jj])

            # previous state (notice that we include both start and end state of the previous time step, because the
            # last row of self.exp_arr_full might be a transition from goal to start):
            last_step_was_goal = self.check_if_goal_step('prev', 'this_goal')
            # last_step_was_goal = np.any(np.isin([self.exp_arr_full[-2, 3], self.exp_arr_full[-2, 0]],
            #                                     np.ravel_multi_index([self.this_goal[:, 0], self.this_goal[:, 1]],
            #                                                          [self.side_ii, self.side_jj])))

            # if we want to plan at the previous goal until the new one is reached
            curr_step_is_old_goal = False
            last_step_was_old_goal = False
            if self.params.plan_at_prev_goal:
                # check whether we are halfway through the experiment (when the goal site changes)
                if self.num_episodes == (self.params.MAX_N_EPISODES / 2):
                    # consider the previous goal as a goal state just for planning
                    curr_step_is_old_goal = self.check_if_goal_step('curr', 'first_goal')
                    # curr_step_is_old_goal = self.exp_arr_full[-1, 3] in np.ravel_multi_index(
                    #     [self.first_goal[:, 0], self.first_goal[:, 1]], [self.side_ii, self.side_jj])

                    # previous state (notice that we include both start and end state of the previous time step,
                    # because the last row of self.exp_arr_full might be a transition from goal to start)
                    last_step_was_old_goal = self.check_if_goal_step('prev', 'first_goal')
                    # last_step_was_old_goal = np.any(np.isin([self.exp_arr_full[-2, 3], self.exp_arr_full[-2, 0]],
                    #                                         np.ravel_multi_index(
                    #                                             [self.first_goal[:, 0], self.first_goal[:, 1]],
                    #                                             [self.side_ii, self.side_jj])))

            if not (curr_step_is_goal or last_step_was_goal or curr_step_is_old_goal or last_step_was_old_goal):
                p = np.Inf  # Otherwise, no planning

        if rew == 0 and self.num_episodes == 0:
            p = np.Inf  # Skip planning before the first reward is encountered

        # Pre-allocate variables to store planning info

        # List of planning backups (to be used for creating a plot with the full planning trajectory/trace)
        planning_backups = np.empty((0, 5))
        backups_gain = []  # List of GAIN for backups executed
        backups_need = []  # List of NEED for backups executed
        backups_EVM = []  # List of EVM for backups executed
        backups_TD = []  # List of (abs(TD)) for backups executed (in case of PS)

        # PLANNING STEPS
        if p == 0:
            plan_start_time = time.perf_counter()
            times_for_gain = 0
            times_for_need = 0
            times_for_EVB = 0
            times_for_PS_prep = 0

        while p < self.params.nPlan:

            if self.replay_strategy in ['EVB', 'gain_only', 'need_only', 'dyna']:

                # Create a list of 1-step backups based on 1-step models
                plan_exp = np.concatenate((np.matlib.repmat(np.arange(self.n_states), 1, self.n_actions).reshape(
                    self.n_actions * self.n_states, 1),
                                           np.repeat(np.arange(self.n_actions), self.n_states, axis=0).reshape(
                                               self.n_actions * self.n_states, 1),
                                           np.column_stack(self.exp_last_rew).reshape(self.n_actions * self.n_states,
                                                                                      1),
                                           np.column_stack(self.exp_last_stp1).reshape(self.n_actions * self.n_states,
                                                                                       1)),
                    axis=1)
                # Remove NaNs -- e.g. actions starting from invalid states, such as walls:
                plan_exp = plan_exp[np.invert(np.isnan(plan_exp).any(axis=1))]
                # Remove actions that lead to same state (optional) -- e.g. hitting the wall:
                if self.params.remove_samestate:
                    plan_exp = plan_exp[plan_exp[:, 0] != plan_exp[:, 3]]
                plan_exp = list(plan_exp)  # use plan_exp to hold all steps of any n-step trajectory

                # Expand previous backup with one extra action
                if self.params.expandFurther and planning_backups.shape[0] > 0:
                    # Find the last entry in planning_backups with that started an n-step backup
                    seq_start = np.argwhere(planning_backups[:, 4] == 1)[-1]
                    seq_so_far = planning_backups[seq_start[0]:, 0:4]
                    sn = int(seq_so_far[-1, 3])  # Final state reached in the last planning step
                    if self.params.onVSoffPolicy == 'on-policy':
                        probs = self.get_act_probs(self.Q[sn])  # Appended experience is sampled on-policy
                    else:
                        probs = np.zeros(self.Q[sn].shape)
                        # Appended experience is sampled greedily:
                        probs[self.Q[sn] == max(self.Q[sn])] = 1 / np.sum(self.Q[sn] == max(self.Q[sn]))
                    #  Select action to append using the same action selection policy used in real experience
                    an = self.probs_to_index(probs)
                    snp1 = self.exp_last_stp1[sn, an]  # Resulting state from taking action an in state sn
                    rn = self.exp_last_rew[sn, an]  # Reward received on this step only
                    next_step_is_nan = np.isnan(snp1) or np.isnan(
                        rn)  # Check whether the retrieved rew and stp1 are NaN
                    # Check whether a loop is formed
                    next_step_is_repeated = np.isin(snp1, [seq_so_far[:, 0], seq_so_far[:, 3]])
                    # p.s. Notice that we can't enforce that planning is done only when the next state is not
                    # repeated or doesn't form a loop. The reason is that the next step needs to be derived
                    # 'on-policy', otherwise the Q-values may not converge.

                    # If loops are  allowed and next state is not repeated, then expand this backup
                    if not next_step_is_nan and (self.params.allowLoops or not next_step_is_repeated):
                        # Add one row to seq_updated (i.e., append one transition). Notice that seq_updated has many
                        # rows, one for each appended step
                        seq_updated = np.append(seq_so_far, np.array([[sn, an, rn, snp1]]), axis=0)
                        plan_exp.append(seq_updated)  # Notice that rew=rn here (only considers reward from this step)

                # Gain term
                if self.replay_strategy in ['EVB', 'gain_only']:
                    gain_start_time = time.perf_counter()
                    gain, sa_gain = self.get_gain(plan_exp)
                    times_for_gain += time.perf_counter() - gain_start_time
                    # self.times_for_gain[self.num_episodes][p - 1] = time.perf_counter() - gain_start_time
                else:
                    gain = list(np.ones((len(plan_exp), 1)))

                # Need term
                if self.replay_strategy in ['EVB', 'need_only']:
                    need_start_time = time.perf_counter()
                    need, SR_or_SD = self.get_need(sti, plan_exp)
                    times_for_need += time.perf_counter() - need_start_time
                    # self.times_for_need[self.num_episodes][p - 1] = time.perf_counter() - need_start_time
                elif self.params.setAllNeedToZero:
                    for e in range(len(plan_exp)):
                        if plan_exp[e][0] == sti:  # Set need to 1 only if updated state is sti
                            need[e] = 1
                        else:
                            need[e] = 0
                    SR_or_SD = np.zeros(SR_or_SD.shape)
                    SR_or_SD[sti] = 1
                else:
                    need = list(np.ones((len(plan_exp), 1)))

                # Expected value of memories

                EVM = np.full((len(plan_exp)), np.nan)
                for i in range(len(plan_exp)):
                    if len(plan_exp[i].shape) == 1:
                        EVM[i] = need[i][-1] * max(gain[i], self.params.baselineGain)
                    elif len(plan_exp[i].shape) == 2:
                        EVM[i] = 0
                        for x in range(len(plan_exp[i])):
                            EVM[i] += need[i][-1] * max(gain[i][-1], self.params.baselineGain)
                    else:
                        err_msg = 'plan_exp[i] does not have the correct shape. It is {} but should have a ' \
                                  'length equal to 1 or 2, e.g. (4,) or (2, 4)'.format(plan_exp[i].shape)
                        raise ValueError(err_msg)

                # PERFORM THE UPDATE
                opport_cost = np.nanmean(self.exp_arr_full[:, 2])  # Average expected reward from a random act
                EVM_thresh = min(opport_cost, self.params.EVMthresh)  # if EVM_thresh==Inf, threshold is opport_cost
                EVB_start_time = time.perf_counter()

                if max(EVM) > EVM_thresh:
                    # Identify state-action pairs with highest priority
                    max_EVM_idx = np.argwhere(EVM == max(EVM))

                    if len(max_EVM_idx) > 1:  # If there are multiple items with equal gain
                        # number of total steps on this trajectory
                        n_steps = np.array([arr.shape[0] if len(arr.shape) > 1 else 1 for arr in plan_exp])
                        if self.params.tieBreak == 'max':
                            # Select the one corresponding to a longer trajectory
                            max_EVM_idx = max_EVM_idx[n_steps[max_EVM_idx] == max(n_steps[max_EVM_idx])]
                        elif self.params.tieBreak == 'min':
                            # Select the one corresponding to a shorter trajectory
                            max_EVM_idx = max_EVM_idx[n_steps[max_EVM_idx] == min(n_steps[max_EVM_idx])]
                        if len(max_EVM_idx) > 1:  # If there are still multiple items with equal gain (and equal length)
                            max_EVM_idx = max_EVM_idx[np.random.randint(len(max_EVM_idx))]  # ... select one at random
                    else:
                        max_EVM_idx = max_EVM_idx[0][0]

                    plan_exp_arr = np.array(plan_exp, dtype=object)
                    if len(plan_exp_arr[max_EVM_idx].shape) == 1:
                        plan_exp_arr_max = np.expand_dims(plan_exp_arr[max_EVM_idx], axis=0)
                    else:
                        plan_exp_arr_max = np.expand_dims(plan_exp_arr[max_EVM_idx][-1], axis=0)

                    for n in range(plan_exp_arr_max.shape[0]):
                        # Retrieve information from this experience
                        s_plan = int(plan_exp_arr_max[n][0])
                        a_plan = int(plan_exp_arr_max[n][1])
                        # Individual rewards from this step to end of trajectory
                        rew_to_end = plan_exp_arr_max[n:][:, 2]
                        # Notice the use of '-1' instead of 'n', meaning that stp1_plan is the final state of the
                        # trajectory
                        stp1_plan = int(plan_exp_arr_max[-1][3])

                        # Discounted cumulative reward from this step to end of trajectory
                        n_plan = np.size(rew_to_end)
                        r_plan = np.dot(np.power(self.params.gamma, np.arange(0, n_plan)), rew_to_end)

                        # ADD PLAN Q_LEARNING UPDATES TO Q_LEARNING FUNCTION
                        if self.params.onVSoffPolicy == 'on-policy':
                            # Learns Qpi -> Expected SARSA(1), or, equivalently, n-step Expected SARSA
                            stp1_value = np.sum(np.multiply(self.Q[stp1_plan], self.get_act_probs(self.Q[stp1_plan])))
                        else:
                            # Learns Q* (can be thought of as 'on-policy' if the target policy is the optimal policy,
                            # since trajectory is sampled greedily)
                            stp1_value = np.max(self.Q[stp1_plan])
                        Q_target = r_plan + (self.params.gamma ** n_plan) * stp1_value
                        if self.params.copyQinPlanBkps:
                            self.Q[s_plan, a_plan] = Q_target
                        else:
                            self.Q[s_plan, a_plan] += self.params.alpha * (Q_target - self.Q[s_plan, a_plan])
                    times_for_EVB += time.perf_counter() - times_for_EVB

                    # self.times_for_EVB[self.num_episodes][p - 1] = time.perf_counter() - times_for_EVB

                    # List of planning backups (to be used in creating a plot with the full planning trajectory/trace)
                    backups_gain.append(gain[max_EVM_idx][0])  # List of GAIN for backups executed
                    backups_need.append(need[max_EVM_idx][0])  # List of NEED for backups executed
                    backups_EVM.append(EVM[max_EVM_idx])  # List of EVM for backups executed

                    if planning_backups.shape[0] > 0:
                        planning_backups = np.vstack(
                            [planning_backups, np.append(plan_exp_arr_max, plan_exp_arr_max.shape[0])])
                    elif planning_backups.shape[0] == 0:
                        planning_backups = np.append(plan_exp_arr_max,
                                                     plan_exp_arr_max.shape[0]).reshape(1, planning_backups.shape[1])
                    else:
                        err_msg = 'planning_backups does not have the correct shape. It is {} but should have a ' \
                                  'length equal to 1 or 2, e.g. (5,) or (2, 5)'.format(planning_backups.shape)
                        raise ValueError(err_msg)
                    p += 1  # Increment planning counter
                else:
                    break

            elif self.replay_strategy == 'prioritized_sweeping':
                PS_prep_time = time.perf_counter()
                self.PQueue.sort(key=lambda exp: exp[0], reverse=True)  # sort PQueue by highest to lowest abs(TD)
                if len(self.PQueue) > self.params.nPlan:
                    self.PQueue = self.PQueue[0:self.params.nPlan]
                times_for_PS_prep += time.perf_counter() - PS_prep_time
                if len(self.PQueue) != 0:
                    this_delta = self.PQueue[0][0]
                    replay_experience = self.PQueue[0][1]  # take first(PQueue) and then the second index: s,a,r,s'
                    self.PQueue.pop(0)  # remove this experience from PQueue
                    s_plan = replay_experience['sti']
                    a_plan = replay_experience['at']
                    r_plan = replay_experience['rew']
                    stp1_plan = replay_experience['stp1i']
                    this_backup = np.array([s_plan, a_plan, r_plan, stp1_plan])

                    ######### ADD PLAN Q_LEARNING UPDATES TO Q_LEARNING FUNCTION
                    if self.params.onVSoffPolicy == 'on-policy':
                        # Learns Qpi -> Expected SARSA(1)
                        stp1_value = np.sum(np.multiply(self.Q[stp1_plan], self.get_act_probs(self.Q[stp1_plan])))
                    else:
                        # Learns Q* (can be thought of as 'on-policy' if the target policy is the optimal policy,
                        # since trajectory is sampled greedily)
                        stp1_value = np.max(self.Q[stp1_plan])

                    Q_target = r_plan + self.params.gamma * stp1_value
                    delta = Q_target - self.Q[s_plan, a_plan]
                    if self.params.copyQinPlanBkps:
                        self.Q[s_plan, a_plan] = Q_target
                    else:
                        self.Q[s_plan, a_plan] += self.params.alpha * delta
                    ########################

                    # identify predecessor states
                    s_plan_sub = np.unravel_index(s_plan, self.params.maze.shape)  # convert to row/column notation
                    ii = s_plan_sub[0]
                    jj = s_plan_sub[1]

                    # state above
                    pre_above = [ii - 1, jj]
                    # state below
                    pre_below = [ii + 1, jj]
                    # state on the right
                    pre_right = [ii, jj + 1]
                    # state on the left
                    pre_left = [ii, jj - 1]

                    predecessors = {'pre_above': {'state': pre_above, 'action': 1},
                                    'pre_below': {'state': pre_below, 'action': 0},
                                    'pre_right': {'state': pre_right, 'action': 3},
                                    'pre_left': {'state': pre_left, 'action': 2}
                                    }

                    ######### ADD TO A FUNCTION - get_state_value
                    if self.params.onVSoffPolicy == 'on-policy':
                        # Learns Qpi -> Expected SARSA(1)
                        s_plan_value = np.sum(np.multiply(self.Q[s_plan], self.get_act_probs(self.Q[s_plan])))
                    else:
                        # Learns Q* (can be thought of as 'on-policy' if the target policy is the optimal policy,
                        # since trajectory is sampled greedily)
                        s_plan_value = np.max(self.Q[s_plan])
                    pre_n = 0
                    for pre_type in predecessors:
                        pre = predecessors[pre_type]
                        pre_s = pre['state']
                        pre_s_x, pre_s_y = pre_s
                        pre_a = pre['action']
                        # update only if predecessor state is within the grid and not a wall state
                        if 0 <= pre_s_x < self.side_ii and 0 <= pre_s_y < self.side_jj and self.params.maze[
                            pre_s_x, pre_s_y] != 1:
                            # convert to an index:
                            pre_si = np.ravel_multi_index(pre_s, self.params.maze.shape)
                            pre_r = self.exp_last_rew[pre_si, pre_a]
                            # get delta of valid predecessors
                            delta = pre_r + self.params.gamma * s_plan_value - self.Q[pre_si, pre_a]
                            self.update_p_queue(delta, pre_si, pre_a, pre_r, s_plan)
                            # increment counter for number of predecessors
                            if not np.isnan(delta):  # no update if delta = nan (e.g. if predecessor is goal state)
                                pre_n += 1

                    backups_TD.append(this_delta)

                    if planning_backups.shape[0] > 0:
                        planning_backups = np.vstack(
                            [planning_backups, np.append(this_backup, pre_n)])
                    elif planning_backups.shape[0] == 0:
                        planning_backups = np.append(this_backup, pre_n).reshape(1, planning_backups.shape[1])
                    else:
                        err_msg = 'planning_backups does not have the correct shape. It is {} but should have a ' \
                                  'length equal to 1 or 2, e.g. (5,) or (2, 5)'.format(planning_backups.shape)
                        raise ValueError(err_msg)

                p += 1  # Increment planning counter

        if planning_backups.shape[0] > 0:  # if planning happened, save in corresponding array

            self.update_performance_df(plan_start_time, p, times_for_PS_prep, times_for_EVB, times_for_gain,
                                       times_for_need)

        return backups_gain, backups_need, backups_EVM, backups_TD, planning_backups

    def update_performance_df(self, plan_start_time, p, times_for_PS_prep, times_for_EVB, times_for_gain,
                              times_for_need):
        """
        transition should be 'g1', 'g2', 'g1_to_2' or 'g2_to_s'
        """
        curr_step_is_g1 = self.check_if_goal_step('curr', 'first_goal')
        curr_step_is_g2 = self.check_if_goal_step('curr', 'second_goal')
        curr_step_is_g1_to_s = self.check_if_goal_step('prev', 'first_goal')
        curr_step_is_g2_to_s = self.check_if_goal_step('prev', 'second_goal')

        if curr_step_is_g1:
            transition = 'g1'
        if curr_step_is_g2:
            transition = 'g2'
        if curr_step_is_g1_to_s:
            transition = 'g1_to_s'
        if curr_step_is_g2_to_s:
            transition = 'g2_to_s'

        var_dict = {'plan_times_': time.perf_counter() - plan_start_time,
                    'n_plan_phases_': 1, 'n_plan_steps_': p}

        if self.replay_strategy == 'prioritized_sweeping':
            var_dict['PS_prep_times_'] = times_for_PS_prep
        if self.replay_strategy in ['EVB', 'gain_only', 'need_only', 'dyna']:
            var_dict['EVB_times_'] = times_for_EVB
        if self.replay_strategy in ['EVB', 'gain_only']:
            var_dict['gain_times_'] = times_for_gain
        if self.replay_strategy in ['EVB', 'need_only']:
            var_dict['need_times_'] = times_for_need

        for key in var_dict:
            val = var_dict[key]
            key += transition
            if np.isnan(self.performance_df[key][self.num_episodes]):
                self.performance_df[key][self.num_episodes] = 0
            self.performance_df[key][self.num_episodes] += val

    def pre_explore_env(self):
        # initialize state indices as a 2-element list denoting the corresponding matrix subscripts (i,j):
        st = [None, None]
        for sti in range(self.n_states):
            for at in range(self.n_actions):
                # Sample action consequences (minus reward, as experiment didn't 'start' yet)
                # Convert state index to matrix subscripts (i,j):
                st[0], st[1] = np.unravel_index(sti, [self.side_ii, self.side_jj])
                # Don't explore walls or goal state (if goal state is included, the agent will be able to replay
                # the experience of performing the various actions at the goal state):
                if (self.params.maze[st[0], st[1]] == 0) and not (st in self.this_goal.tolist()):
                    # return index at state plus one (successor state):
                    _, _, stp1i = self.sample_transition(st, at)
                    # Update list of experiences:
                    self.exp_arr_full = np.append(self.exp_arr_full, [[sti, at, 0, stp1i]], axis=0)  # 0 reward
                    self.exp_last_stp1[sti, at] = stp1i  # stp1 from last experience of this state/action
                    self.exp_last_rew[sti, at] = 0  # rew from last experience of this state/action
                    self.T[sti, stp1i] += 1  # Update transition matrix
        self.build_transition_mat()

    def build_transition_mat(self):
        # normalize so that rows sum to one (ie build a proper transition function)
        t_col = np.expand_dims(np.nansum(self.T, axis=1), axis=1)
        np.seterr(invalid='ignore')  # ignore warning about dividing by 0 (will just give nan values)
        self.T = np.divide(self.T, np.matlib.repmat(t_col, 1, self.T.shape[0]))
        np.seterr(invalid='warn')  # bring back warning so that it can be flagged in cases that are not this exception
        self.T[np.isnan(self.T)] = 0  # nans correspond to transitions from wall states, so set equal to 0
        # Add transitions from goal states to start states
        if self.params.Tgoal2start:
            for i in range(self.this_goal.shape[0]):  # Loop through each goal state
                if not self.params.s_start_rand:  # if starting locations are not randomized
                    gi = np.ravel_multi_index(self.this_goal[i], [self.side_ii, self.side_jj])  # goal state index
                    # beginning state index:
                    bi = np.ravel_multi_index(self.params.s_start[0], [self.side_ii, self.side_jj])
                    self.T[gi] = 0  # Transitions from goal to anywhere else: 0
                    self.T[gi, bi] = 1  # Transitions from goal to start: 1
                else:  # if starting locations are randomized
                    # goal state index:
                    gi = np.ravel_multi_index(
                        [self.this_goal[i, 0], self.this_goal[i, 1]], [self.side_ii, self.side_jj])
                    # transitions from goal to anywhere else equals 0:
                    self.T[gi] = 0
                    # transitions from this goal to anywhere else is uniform:
                    valid_states = self.get_valid_states()
                    self.T[gi, valid_states] = 1 / len(self.T[gi, valid_states])

    def explore_env(self):

        # PREPARE FIRST TRIAL
        # Move the agent to the (first) starting state
        st, sti = self.get_starting_state()

        # plot maze image with agent
        self.plot_agent(st)

        # EXPLORE MAZE
        ts = 0  # initialize number of time-steps
        print('\nStarting a new simulation (exploring the environment); strategy: {}'.format(self.replay_strategy))
        start_time = time.perf_counter()  # start timer/counter
        for tsi in range(self.params.MAX_N_STEPS):
            # Display progress
            progress = '{} steps; {} episodes'.format(tsi, self.num_episodes)
            print(progress, end='\r')

            # ACTION SELECTION
            at = self.sample_action(sti)

            # PERFORM ACTION
            # move to state stp1 and collect reward
            rew, stp1, stp1i = self.sample_transition(st, at)  # state and action to state plus one and reward

            # UPDATE TRANSITION MATRIX AND EXPERIENCE LIST
            self.update_transition_mat(sti, stp1i)
            # add transition to self.exp_arr_full:
            self.exp_arr_full = np.append(self.exp_arr_full, [[sti, at, rew, stp1i]], axis=0)
            self.exp_arr = np.append(self.exp_arr, [[sti, at, rew, stp1i]], axis=0)
            self.exp_last_stp1[sti, at] = stp1i  # stp1 from last experience of this state/action
            self.exp_last_rew[sti, at] = rew  # reward from last experience of this state/action

            # UPDATE Q-VALUES (Q-LEARNING)
            self.Q_learning(sti, at, rew, stp1i)

            # PLAN
            # if self.params.nPlan > 0 or self.replay_strategy != 'no_replay':
            backups_gain, backups_need, backups_EVM, backups_TD, planning_backups = self.plan(rew, sti)

            # MOVE AGENT TO NEXT STATE
            st = stp1
            sti = stp1i

            # update maze image with agent
            self.plot_agent(st)
            # plot Q values
            self.plot_Q_map()

            # COMPLETE STEP
            ts += 1  # Timesteps to solution (reset to zero at the end of the episode)
            if st in self.this_goal.tolist():  # Agent is at a terminal state
                if self.params.s_start_rand:
                    # Pick next start state at random
                    stp1, stp1i = self.get_starting_state()
                else:
                    # Determine which of the possible start states to use
                    goal_num = np.argwhere(np.all(np.isin(self.this_goal, st), axis=0))[0][0]
                    startnum = goal_num % self.params.s_start.shape[0]
                    stp1 = self.params.s_start[startnum]
                    stp1i = np.ravel_multi_index(stp1, [self.side_ii, self.side_jj])
                    self.this_starting_state_i = stp1i

                # Update transition matrix and list of experiences
                if self.params.Tgoal2start:
                    target_vec = np.zeros((1, self.n_states))
                    target_vec[0][stp1i] = 1
                    # Shift corresponding row of T towards target_vec
                    self.T[sti] += (self.params.TLearnRate * (target_vec - self.T[sti]))[0]
                    # Add transition to expList:
                    self.exp_arr_full = np.append(self.exp_arr_full, [[sti, np.nan, np.nan, stp1i]], axis=0)
                # Move the agent to the next location
                st = stp1
                sti = stp1i
                # record elapsed time during episode + planning
                stop_time = time.perf_counter()
                self.performance_df['full_time_per_episode'][self.num_episodes] = stop_time - start_time
                # record "ts" timesteps that it took the agent to get to the solution (end state)
                self.performance_df['steps_per_episode'][self.num_episodes] = ts
                ts = 0
                self.elig_trace = np.zeros(self.elig_trace.shape)  # Reset eligibility matrix
                self.num_episodes += 1  # Record that we got to the end
                if self.params.change_goal:
                    # halfway through the experiment, change the goal location:
                    if self.num_episodes == np.ceil(self.params.MAX_N_EPISODES / 2):
                        self.last_tsi_at_goal_1 = tsi
                        self.this_goal = self.params.s_end_change
                        print("\nGoal location changed")
                    # the next episode means that the agent found the new goal; record this
                    if self.num_episodes == self.params.MAX_N_EPISODES / 2 + 1:
                        self.first_tsi_at_goal_2 = tsi
                start_time = time.perf_counter()  # reset start time

            # SAVE SIMULATION DATA
            self.num_episodes_arr[tsi] = self.num_episodes
            assert self.exp_arr.shape[0] == tsi + 1, 'self.exp_arr has incorrect size'
            if planning_backups.shape[0] > 0:  # If there was planning in this timestep
                if planning_backups.shape[0] < 20:  # if less than 20 planning steps (e.g. priotitized sweeping)
                    n_missing_rows = 20 - planning_backups.shape[0]
                    # fill missing rows with nans
                    planning_backups = np.vstack([planning_backups, np.full((n_missing_rows, 5), np.nan)])
                # In a multi-step sequence, self.replay['state'] has 1->2 in one row, 2->3 in another row, etc
                self.replay['state'][tsi] = planning_backups[:, 0]
                self.replay['action'][tsi] = planning_backups[:, 1]
                backups = {'gain': backups_gain, 'need': backups_need, 'EVM': backups_EVM, 'TD': backups_TD}
                for key in backups:
                    backup = backups[key]
                    if len(backup) > 0:  # if not empty
                        if len(backup) < 20:
                            n_missing_rows = 20 - len(backup)
                            for i in range(n_missing_rows):
                                backup.append(np.nan)
                        self.replay[key][tsi] = backup

            # If max number of episodes is reached, trim down simData.replay
            if self.num_episodes == self.params.MAX_N_EPISODES:
                self.num_episodes_arr = self.num_episodes_arr[0:tsi + 1]
                self.replay['state'] = self.replay['state'][0:tsi + 1]
                self.replay['action'] = self.replay['action'][0:tsi + 1]
                self.replay['gain'] = self.replay['gain'][0:tsi + 1]
                self.replay['need'] = self.replay['need'][0:tsi + 1]
                self.replay['EVM'] = self.replay['EVM'][0:tsi + 1]
                self.replay['TD'] = self.replay['TD'][0:tsi + 1]
                # msg = 'END OF SIMULATION\n{} steps; {} episodes'.format(tsi, self.num_episodes)
                # print('\r', msg, end='')
                break
