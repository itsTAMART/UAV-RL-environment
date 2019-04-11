# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
import sys
import os

from stable_baselines import ACKTR
from stable_baselines.common.policies import MlpPolicy

from uav_enviroment.UAV_Environment import UAVEnv
from uav_enviroment.curriculum_learning_UAVenv import Curriculum_UAVEnv
from utils.evaluate_model import *
from utils.logs_callback import *

SEED = 16
EXPERIMENT_NATURE = 'UAVenv_discrete_cartesian'


def setup_env_cart_discrete(seed, log_dir):
    """
    Sets up the environment with cartesian observations and discrete action space.
    :param seed: random seed
    :return: the environment
    """
    env = UAVEnv(continuous=False, angular_movement=False, observation_with_image=False, reset_always=True,
                 controlled_speed=False)
    env.setup(n_obstacles=6, reset_always=True, threshold_dist=20, reward_sparsity=True)
    env.seed(seed)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    return env


def setup_env_curriculum_learning(seed, log_dir):
    """
    Sets up the environment with cartesian observations and discrete action space.
    :param seed: random seed
    :return: the environment
    """
    env = Curriculum_UAVEnv(continuous=False, angular_movement=False, observation_with_image=False, reset_always=True,
                            controlled_speed=False)
    env.setup(n_obstacles=0, reset_always=True, threshold_dist=60, reward_sparsity=True)
    env.seed(seed)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    return env


def experiment_ACKTR():
    pass


if __name__ == '__main__':
    exp_type = sys.argv[1]
    seed = sys.argv[2]

    print('Starting experiment of type {} and seed {}'.format(exp_type, seed))
