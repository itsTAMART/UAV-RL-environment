import os

import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.signal as sp

from matplotlib2tikz import save as tikz_save

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DDPG
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec

global n_steps, best_mean_reward
best_mean_reward, n_steps = -np.inf, 0
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)


def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward, log_dir

  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              print(x[-1], 'timesteps')
              print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward,
                                                                                             mean_reward))
              globals()['best_mean_reward'] = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(log_dir + 'best_model.pkl')
  n_steps += 1
  return True


def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_average(values, window):
    return np.mean(rolling_window(values, window), -1)


def rolling_std(values, window):
    return np.std(rolling_window(values, window), -1)


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    plt.figure(title)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.ylim([-5, 0.5])
    plt.xlim([0, 3000000])
    plt.title(title + " Smoothed")
    for experiment_run in os.listdir(log_folder):
        df = pd.read_csv(log_folder + '/' + experiment_run, skiprows=1)
        x = df['l'].cumsum()
        y = df['r']
        y = movingAverage(y, window=700)
        # Truncate x
        print(title + '{:.5f} '.format(y[-60]) + experiment_run)
        x = x[len(x) - len(y):]
        plt.plot(x, y, label=experiment_run.replace(".csv", ""))

    plt.hlines(0, xmin=0, xmax=np.max(x), linestyles='dashed')
    plt.legend()
    plt.show()


def paper_plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    window_size = 400
    decimate_factor = 100

    plt.figure(title)
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.ylim([-9, 0.5])
    plt.xlim([0, 980000])
    # plt.title(title)
    for experiment_run in os.listdir(log_folder):
        df = pd.read_csv(log_folder + '/' + experiment_run, skiprows=1)
        x = df['l'].cumsum()
        y = df['r']
        # y = movingAverage(y, window=window_size)

        y_mean = rolling_average(y, window=window_size)
        y_std = rolling_std(y, window=window_size)
        x = x[:min(len(x), len(y_mean))]

        # # Decimating
        # x = sp.decimate(x,decimate_factor)
        # y_mean = sp.decimate(y_mean,decimate_factor)
        # y_std = sp.decimate(y_std,decimate_factor)

        # Truncate x
        # print(title+'{:.5f} '.format(y_mean[-60])+experiment_run)
        # print(len(x))
        # print(len(y_mean))

        plt.plot(x, y_mean, label=experiment_run.replace(".csv", ""))
        plt.fill_between(x, y1=y_mean - 0.5 * y_std, y2=y_mean + 0.5 * y_std, alpha=0.1)

    plt.hlines(0, xmin=0, xmax=np.max(x), linestyles='dashed')
    plt.legend()
    # tikz_save('comparison_3M.tikz', figureheight='\\figureheight', figurewidth='\\figurewidth')
    plt.show()



def plot_experiment(experiment='UAVenv_discrete_cartesian'):
    csv_dir = './logs/{}/csv'.format(experiment)
    for i, experiment in enumerate(os.listdir(csv_dir)):
        plot_results(csv_dir + '/' + experiment, title=experiment)


def paper_plot_experiment(experiment='UAVenv_discrete_cartesian'):
    csv_dir = '{}/csv'.format(experiment)

    experiment = 'Comparison'
    paper_plot_results(csv_dir + '/' + experiment, title=experiment)
