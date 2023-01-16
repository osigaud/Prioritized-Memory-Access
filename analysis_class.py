import pickle
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import warnings
from main import Replay_Sim
import math
import statistics


class Sim_Data:

    def __init__(self, maze, models, print_info=False):
        self.maze = maze
        self.models = models
        self.model_data, self.params_dict = self.extract_data(print_info)

    def extract_data(self, print_info):
        model_data = {}
        params_dict = {}
        for i, model in enumerate(self.models):
            model_data[model] = {}
            file_list = [file for file in glob.glob(os.path.join('checkpoints', self.maze, model + '*'))]
            #print("LLOOOKK"+ str(file_list))
            for k, file in enumerate(file_list):
                with open(file, 'rb') as f:
                    model_data[model][k] = pickle.load(f)
                if i == 0:
                    params_dict = model_data[model][k].params_dict
                if print_info:
                    print('maze: ', self.maze)
                    for key in params_dict:
                        print(key, ': ', params_dict[key])
        return model_data, params_dict


    def plot_fig(self, dependent_var, y_label=None, title=None, log_values=False, crop_y=False):
        fig = plt.figure()
        fig_title = '{} maze: '.format(self.maze).replace("_", " ").title() + title
        if y_label is None:
            y_label = dependent_var
        if log_values:
            y_label += ' (log)'
            fig_title += ' (log)'
        plt.title(fig_title)
        plt.ylabel(y_label)
        if crop_y and not log_values:
            plt.ylim(0, 200)
        plt.xlabel('# Episodes')
        plt.xticks(np.arange(0, self.params_dict['MAX_N_EPISODES'] + 1, 5))
        plt.xlim(1, self.params_dict['MAX_N_EPISODES'])
        plt.axvline(self.params_dict['MAX_N_EPISODES'] / 2, linestyle=':', color='gray', label='goal change')

        dependent_var_dict = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for model in self.models:
                dependent_var_dict[model] = np.empty(
                    (self.params_dict['N_SIMULATIONS'], self.params_dict['MAX_N_EPISODES']))
                for k in range(self.params_dict['N_SIMULATIONS']):
                    if dependent_var in ['steps_per_episode', 'full_time_per_episode']:
                        dependent_var_dict[model][k] = self.model_data[model][k].performance_df[dependent_var]
                    else:
                        transitions = ['_g1', '_g2', '_g1_to_s', '_g2_to_s']
                        dependent_var_dict[model][k] = np.zeros(self.params_dict['MAX_N_EPISODES'])
                        for tr in transitions:
                            this_var = dependent_var + tr
                            dependent_var_dict[model][k] = np.nansum(np.dstack(
                                (dependent_var_dict[model][k], self.model_data[model][k].performance_df[this_var])), 2)
                if log_values:
                    means = np.nanmean(np.log(dependent_var_dict[model]), axis=0)
                    stds = np.std(np.log(dependent_var_dict[model]), axis=0) / np.sqrt(
                        self.params_dict['N_SIMULATIONS'])
                else:
                    means = np.nanmean(dependent_var_dict[model], axis=0)
                    stds = np.std(dependent_var_dict[model], axis=0) / np.sqrt(self.params_dict['N_SIMULATIONS'])
                # plot figure for model across all simulations
                plt.plot(range(1, 1 + len(means)), means, label=model)
                plt.fill_between(range(1, 1 + len(means)), means - stds, means + stds, alpha=0.3)
        plt.legend(loc="upper right")
        file_name = fig_title.replace(" ", "_").replace(":", "")
        maze_path = os.path.join('checkpoints', self.maze)
        maze_fig_path = os.path.join(maze_path, 'figures')
        if 'figures' not in os.listdir(maze_path):
            os.mkdir(maze_fig_path)
        fig.savefig(maze_fig_path + '/{}.png'.format(file_name), dpi=2000)
