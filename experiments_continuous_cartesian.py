# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy

from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import PPO2
from stable_baselines import TRPO
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy

from uav_enviroment.UAV_Environment import UAVEnv
from utils.evaluate_model import *
from utils.logs_callback import *

SEED = 16
NUM_CPU = 2
EXPERIMENT_NATURE = 'UAVenv_continuous_cartesian'


def mock_up_test(name):
    print('Training {}...'.format(name))
    n = np.random.randn()
    time.sleep((n ** 2) / 6)
    evaluation = evaluate_model(None, None, 100)
    score = n ** 2

    return score, evaluation


# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)


def setup_env_cart_continuous(seed):
    """
    Sets up the environment with cartesian observations and discrete action space.
    :param seed: random seed
    :return: the environment
    """
    env = UAVEnv(continuous=True, angular_movement=False, observation_with_image=False, reset_always=True)
    env.setup(n_obstacles=2, reset_always=True, threshold_dist=20)
    env.seed(seed)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    return env


set_up_env = setup_env_cart_continuous


# # (name, parametrized_model, method)
# experiments = [
#     ('DDPG',
#      DDPG(),
#      DDPG),
#     ('A2C',
#      A2C(policy=MlpPolicy, env=env, gamma=0.99, n_steps=5, vf_coef=0.25,
#                 ent_coef=0.01, max_grad_norm=0.5, learning_rate=0.0007, alpha=0.99,
#                 epsilon=1e-05, lr_schedule='linear', verbose=0,
#                 tensorboard_log="./logs/{}/tensorboard/{}/".format(EXPERIMENT_NATURE, algo)),
#      A2C),
#     ('TRPO',
#       TRPO(policy=MlpPolicy, env=env, gamma=0.99, timesteps_per_batch=128,
#                  max_kl=0.01, cg_iters=10, lam=0.98, entcoeff=0.0, cg_damping=0.01,
#                  vf_stepsize=0.0003, vf_iters=3, verbose=0,
#                  tensorboard_log="./logs/{}/tensorboard/{}/".format(EXPERIMENT_NATURE, algo)),
#     TRPO),
#     ('PPO2',
#      PPO2(policy=MlpPolicy, env=env, gamma=0.99, n_steps=512, ent_coef=0.01,
#           learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, lam=0.95,
#           nminibatches=1, noptepochs=4, cliprange=0.2, verbose=0,
#           tensorboard_log="./logs/{}/tensorboard/{}/".format(EXPERIMENT_NATURE, algo)),
#      PPO2)
# ]
#
# def train_model(name, parametrized_model, method, seed):
#     """
#     Generic call for training and evaluating a model
#
#     :param name: method name e.g. 'DQN'
#     :param parametrized_model: method call with all the parameters e.g. DQN(n_t=....)
#     :param method: method used to call load from e.g. DQN
#     :param seed: random seed
#     :return: the evaluation from evaluate_model
#     """
#     algo = name
#     num_timesteps = 3000000
#
#     env = set_up_env(seed)
#
#     global best_mean_reward, n_steps
#     best_mean_reward, n_steps = -np.inf, 0
#
#     # Tested with n_steps=128
#     model = parametrized_model
#
#     model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed,
#                 log_interval=500, tb_log_name="seed_{}".format(seed))
#
#     model = method.load(log_dir + 'best_model.pkl')
#
#     evaluation = evaluate_model(env, model, 100)
#     os.makedirs('./logs/{}/csv/{}/'.format(EXPERIMENT_NATURE, algo), exist_ok=True)
#     os.rename('/tmp/gym/monitor.csv', "./logs/{}/csv/{}/seed_{}.csv".format(EXPERIMENT_NATURE, algo, seed))
#     env.close()
#     del model, env
#     gc.collect()
#     return evaluation
#
#

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
# parametrize("policy", ['cnn', 'lstm', 'lnlstm'])
def train_a2c(seed):
    """
    test A2C on the uav_env(cartesian,discrete) 
    :param seed: (int) random seed for A2C
    """
    """
    A2C(policy, env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, 
    max_grad_norm=0.5, learning_rate=0.0007, alpha=0.99, epsilon=1e-05,
    lr_schedule='linear', verbose=0,tensorboard_log=None, _init_setup_model=True)
    """
    algo = 'A2C'
    num_timesteps = 3000000

    env = set_up_env(seed)

    global best_mean_reward, n_steps
    best_mean_reward, n_steps = -np.inf, 0

    model = A2C(policy=MlpPolicy, env=env, gamma=0.99, n_steps=5, vf_coef=0.25,
                ent_coef=0.01, max_grad_norm=0.5, learning_rate=0.0007, alpha=0.99,
                epsilon=1e-05, lr_schedule='linear', verbose=0,
                tensorboard_log="./logs/{}/tensorboard/{}/".format(EXPERIMENT_NATURE, algo))

    model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed,
                log_interval=500, tb_log_name="seed_{}".format(seed))

    model = A2C.load(log_dir + 'best_model.pkl')

    evaluation = evaluate_model(env, model, 100)
    os.makedirs('./logs/{}/csv/{}/'.format(EXPERIMENT_NATURE, algo), exist_ok=True)
    os.rename('/tmp/gym/monitor.csv', "./logs/{}/csv/{}/seed_{}.csv".format(EXPERIMENT_NATURE, algo, seed))
    env.close()
    del model, env
    gc.collect()
    return evaluation


from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
def train_a2c_recurrent(seed):
    """
    test A2C on the uav_env(cartesian,discrete)
    :param seed: (int) random seed for A2C
    """
    """
    A2C(policy, env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, 
    max_grad_norm=0.5, learning_rate=0.0007, alpha=0.99, epsilon=1e-05,
    lr_schedule='linear', verbose=0,tensorboard_log=None, _init_setup_model=True)
    """
    algo = 'A2C_recurrent'
    num_timesteps = 3000000

    env = set_up_env(seed)

    global best_mean_reward, n_steps
    best_mean_reward, n_steps = -np.inf, 0

    model = A2C(policy=MlpLstmPolicy, env=env, gamma=0.99, n_steps=5, vf_coef=0.25,
                ent_coef=0.01, max_grad_norm=0.5, learning_rate=0.0007, alpha=0.99,
                epsilon=1e-05, lr_schedule='linear', verbose=0,
                tensorboard_log="./logs/{}/tensorboard/{}/".format(EXPERIMENT_NATURE, algo))

    model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed,
                log_interval=500, tb_log_name="seed_{}".format(seed))

    # model = A2C.load(log_dir + 'best_model.pkl')

    evaluation = evaluate_model(env, model, 100)
    os.makedirs('./logs/{}/csv/{}/'.format(EXPERIMENT_NATURE, algo), exist_ok=True)
    os.rename('/tmp/gym/monitor.csv', "./logs/{}/csv/{}/seed_{}.csv".format(EXPERIMENT_NATURE, algo, seed))
    env.close()
    del model, env
    gc.collect()
    return evaluation


def train_ddpg(seed):
    """
    test DDPG on the uav_env(cartesian,discrete)
    :param seed: (int) random seed for A2C
    """
    """
    DDPG(policy, env, gamma=0.99, memory_policy=None, eval_env=None, 
    nb_train_steps=50, nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, 
    action_noise=None, normalize_observations=False, tau=0.001, batch_size=128, 
    param_noise_adaption_interval=50, normalize_returns=False, enable_popart=False, 
    observation_range=(-5.0, 5.0), critic_l2_reg=0.0, return_range=(-inf, inf), 
    actor_lr=0.0001, critic_lr=0.001, clip_norm=None, reward_scale=1.0, render=False, 
    render_eval=False, memory_limit=100, verbose=0, 
    tensorboard_log=None, _init_setup_model=True)
    """
    algo = 'DDPG'
    num_timesteps = 3000000

    env = set_up_env(seed)

    global best_mean_reward, n_steps
    best_mean_reward, n_steps = -np.inf, 0

    model = DDPG(policy=DDPGMlpPolicy, env=env, gamma=0.99, memory_policy=None,
                 eval_env=None, nb_train_steps=50, nb_rollout_steps=100,
                 nb_eval_steps=100, param_noise=None, action_noise=None,
                 normalize_observations=False, tau=0.001, batch_size=128,
                 param_noise_adaption_interval=50, normalize_returns=False,
                 enable_popart=False, observation_range=(-5.0, 5.0), critic_l2_reg=0.0,
                 actor_lr=0.0001, critic_lr=0.001, clip_norm=None, reward_scale=1.0,
                 render=False, render_eval=False, memory_limit=100, verbose=0,
                 tensorboard_log="./logs/{}/tensorboard/{}/".format(EXPERIMENT_NATURE, algo))

    model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed,
                log_interval=500, tb_log_name="seed_{}".format(seed))

    # model = DDPG.load(log_dir + 'best_model.pkl')

    evaluation = evaluate_model(env, model, 100)
    os.makedirs('./logs/{}/csv/{}/'.format(EXPERIMENT_NATURE, algo), exist_ok=True)
    os.rename('/tmp/gym/monitor.csv', "./logs/{}/csv/{}/seed_{}.csv".format(EXPERIMENT_NATURE, algo, seed))
    env.close()
    del model, env
    gc.collect()
    return evaluation


def train_trpo(seed):
    """
    test TRPO on the uav_env(cartesian,discrete)
    """
    """
    TRPO(policy, env, gamma=0.99, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, 
    lam=0.98, entcoeff=0.0, cg_damping=0.01, vf_stepsize=0.0003, vf_iters=3, verbose=0, 
    tensorboard_log=None, _init_setup_model=True)
    """
    algo = 'TRPO'
    num_timesteps = 3000000

    env = set_up_env(seed)

    global best_mean_reward, n_steps
    best_mean_reward, n_steps = -np.inf, 0

    # Tested with: timesteps_per_batch=1024
    model = TRPO(policy=MlpPolicy, env=env, gamma=0.99, timesteps_per_batch=128,
                 max_kl=0.01, cg_iters=10, lam=0.98, entcoeff=0.0, cg_damping=0.01,
                 vf_stepsize=0.0003, vf_iters=3, verbose=0,
                 tensorboard_log="./logs/{}/tensorboard/{}/".format(EXPERIMENT_NATURE, algo))

    model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed,
                log_interval=500, tb_log_name="seed_{}".format(seed))

    # model = TRPO.load(log_dir + 'best_model.pkl')

    evaluation = evaluate_model(env, model, 100)
    os.makedirs('./logs/{}/csv/{}/'.format(EXPERIMENT_NATURE, algo), exist_ok=True)
    os.rename('/tmp/gym/monitor.csv', "./logs/{}/csv/{}/seed_{}.csv".format(EXPERIMENT_NATURE, algo, seed))
    env.close()
    del model, env
    gc.collect()
    return evaluation


# parametrize("policy", ['cnn', 'lstm', 'lnlstm', 'mlp'])
def train_ppo2(seed):
    """
    test PPO2 on the uav_env(cartesian,discrete)
    :param seed:
    :return:
    """
    """
    PPO2(policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=0.00025, 
    vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, 
    cliprange=0.2, verbose=0, tensorboard_log=None, _init_setup_model=True)
    """
    algo = 'PPO2'
    num_timesteps = 3000000

    env = set_up_env(seed)

    global best_mean_reward, n_steps
    best_mean_reward, n_steps = -np.inf, 0

    # Tested with n_steps=128
    model = PPO2(policy=MlpPolicy, env=env, gamma=0.99, n_steps=512, ent_coef=0.01,
                 learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, lam=0.95,
                 nminibatches=1, noptepochs=4, cliprange=0.2, verbose=0,
                 tensorboard_log="./logs/{}/tensorboard/{}/".format(EXPERIMENT_NATURE, algo))

    model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed,
                log_interval=500, tb_log_name="seed_{}".format(seed))

    # model = PPO2.load(log_dir + 'best_model.pkl')

    evaluation = evaluate_model(env, model, 100)
    os.makedirs('./logs/{}/csv/{}/'.format(EXPERIMENT_NATURE, algo), exist_ok=True)
    os.rename('/tmp/gym/monitor.csv', "./logs/{}/csv/{}/seed_{}.csv".format(EXPERIMENT_NATURE, algo, seed))
    env.close()
    del model, env
    gc.collect()
    return evaluation


# parametrize("policy", ['cnn', 'lstm', 'lnlstm', 'mlp'])
def train_ppo2_recurrent(seed):
    """
    test PPO2 on the uav_env(cartesian,discrete)
    :param seed:
    :return:
    """
    """
    PPO2(policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=0.00025, 
    vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, 
    cliprange=0.2, verbose=0, tensorboard_log=None, _init_setup_model=True)
    """
    algo = 'PPO2_recurrent'
    num_timesteps = 3000000

    env = set_up_env(seed)

    global best_mean_reward, n_steps
    best_mean_reward, n_steps = -np.inf, 0

    # Tested with n_steps=128
    model = PPO2(policy=MlpLstmPolicy, env=env, gamma=0.99, n_steps=512, ent_coef=0.01,
                 learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5, lam=0.95,
                 nminibatches=1, noptepochs=4, cliprange=0.2, verbose=0,
                 tensorboard_log="./logs/{}/tensorboard/{}/".format(EXPERIMENT_NATURE, algo))

    model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed,
                log_interval=500, tb_log_name="seed_{}".format(seed))

    # model = PPO2.load(log_dir + 'best_model.pkl')

    evaluation = evaluate_model(env, model, 100)
    os.makedirs('./logs/{}/csv/{}/'.format(EXPERIMENT_NATURE, algo), exist_ok=True)
    os.rename('/tmp/gym/monitor.csv', "./logs/{}/csv/{}/seed_{}.csv".format(EXPERIMENT_NATURE, algo, seed))
    env.close()
    del model, env
    gc.collect()
    return evaluation
