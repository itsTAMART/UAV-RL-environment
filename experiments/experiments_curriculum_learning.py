# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
import sys
import os


from uav_enviroment.UAV_Environment import UAVEnv
from uav_enviroment.curriculum_learning_UAVenv import Curriculum_UAVEnv
from utils.evaluate_model import *
from utils.logs_callback import *

from stable_baselines import ACKTR
from stable_baselines.common.policies import MlpPolicy


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


num_timesteps = 300000
best_mean_reward, n_steps = -np.inf, 0

if __name__ == '__main__':
    exp_type = sys.argv[1]  # 'curriculum' for curriculum learning
    experiment_dir = sys.argv[2]
    seed = int(sys.argv[3])

    # Create the folder for the seed
    print('Creating log folders')
    log_dir = experiment_dir + '/{}_seed_{}/'.format(exp_type, seed)
    os.makedirs(log_dir, exist_ok=True)
    model_dir = log_dir + '/models/'
    os.makedirs(model_dir, exist_ok=True)
    trajectory_dir = log_dir + '/trajectories/'
    os.makedirs(trajectory_dir, exist_ok=True)

    print('Starting experiment of type {} and seed {} in log_dir {}'.format(exp_type, seed, log_dir))

    if exp_type == 'curriculum':
        set_up_env = setup_env_curriculum_learning
    else:
        set_up_env = setup_env_cart_discrete

    env = set_up_env(seed, log_dir)

    best_mean_reward, n_steps = -np.inf, 0


    def custom_callback(_locals, _globals):
        """
        Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
        :param _locals: (dict)
        :param _globals: (dict)
        """
        global n_steps, best_mean_reward, log_dir

        # Print stats every 1000 calls
        if (n_steps + 1) % 1000 == 0:
            # seed = _locals['seed']
            # experiment_dir = log_dir + 'seed_{}/'.format(seed)
            # Evaluate policy performance
            x, y = ts2xy(load_results(log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])

                # New best model, you could save the agent here
                if mean_reward > best_mean_reward:
                    print(x[-1], 'timesteps')
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(mean_reward,
                                                                                                   best_mean_reward))
                    globals()['best_mean_reward'] = mean_reward
                    # Example for saving best model
                    print("Saving new best model")
                    _locals['self'].save(log_dir + 'best_model_{}_steps.pkl'.format(n_steps + 1))

        if (n_steps + 1) % 100000 == 0:
            print("Saving checkpoint model")
            _locals['self'].save(model_dir + 'model_{}_steps.pkl'.format(n_steps + 1))

        n_steps += 1
        return True


    print('Starting Training')
    """
    ACKTR(policy, env, gamma=0.99, nprocs=1, n_steps=20, ent_coef=0.01, vf_coef=0.25,
     vf_fisher_coef=1.0, learning_rate=0.25, max_grad_norm=0.5, kfac_clip=0.001,
      lr_schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True,
       async_eigen_decomp=False, policy_kwargs=None, full_tensorboard_log=False)
    """

    model = ACKTR(policy=MlpPolicy, env=env, gamma=0.99, nprocs=1, n_steps=20,
                  ent_coef=0.01, vf_coef=0.25, vf_fisher_coef=1.0, learning_rate=0.25,
                  max_grad_norm=0.5, kfac_clip=0.001, lr_schedule='linear', verbose=0,
                  tensorboard_log=None, _init_setup_model=True)

    model.learn(total_timesteps=num_timesteps, callback=custom_callback, seed=seed,
                log_interval=100)

    print('Starting evaluation')
    env = setup_env_cart_discrete(seed, log_dir)
    model.set_env(env)

    get_trajectories(model, trajectory_dir, n_trajectories=100)
