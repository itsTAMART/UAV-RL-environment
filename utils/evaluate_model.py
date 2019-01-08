import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


def evaluate_model(env, model, n_episodes=50, render_eval=False):
    """
    Evaluates the agent on a set number of episodes.
    :param env: environment
    :param model: model to test
    :param n_episodes:
    :param render_eval: True to render the process
    :return: the results of the evaluation
    """
    model.set_env(env)

    print('Evaluating Model...')
    n_good, n_timeout, n_bad, avg_good_ep_len, avg_bad_ep_len = 0, 0, 0, -1, -1

    GAMMA = .99

    episode_rewards = []
    good_ep_len = []
    bad_ep_len = []
    for i in range(n_episodes):
        cumulative_reward = 0
        done = [False]
        obs = env.reset()
        timestep = 0
        while not done[0]:
            timestep += 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step([action])
            cumulative_reward = rewards[0] + GAMMA * cumulative_reward
            if render_eval and (timestep % 5 == 0 or done[0]):
                env.envs[0].render()

        episode_rewards.append(cumulative_reward)

        if timestep == model.get_env().unwrapped.envs[0].unwrapped.max_timestep:
            n_timeout += 1
        elif rewards[0] > 0 :
            good_ep_len.append(timestep)
            n_good += 1
        else:
            bad_ep_len.append(timestep)
    print('{}/{} successful episodes'.format(n_good,n_episodes))

    # Ensuring the mean doesn't break the program
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        score = np.mean(episode_rewards)
        avg_good_ep_len = np.mean(good_ep_len)
        avg_bad_ep_len = np.mean(bad_ep_len)

    n_bad = n_episodes - n_good - n_timeout

    return [score, n_good, n_timeout, n_bad, avg_good_ep_len, avg_bad_ep_len]
