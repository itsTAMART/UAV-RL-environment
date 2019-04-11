# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy

from stable_baselines import A2C
from stable_baselines import ACER
from stable_baselines import ACKTR
from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines import TRPO
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy

from uav_enviroment.UAV_Environment import UAVEnv
from utils.evaluate_model import *
from utils.logs_callback import *

SEED = 16
NUM_CPU = 2
EXPERIMENT_NATURE = 'UAVenv_discrete_cartesian'


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


def setup_env_cart_discrete(seed):
    """
    Sets up the environment with cartesian observations and discrete action space.
    :param seed: random seed
    :return: the environment
    """
    env = UAVEnv(continuous=False, angular_movement=False, observation_with_image=False, reset_always=True)
    env.setup(n_obstacles=2, reset_always=True, threshold_dist=20)
    env.seed(seed)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    return env


set_up_env = setup_env_cart_discrete


def train_deepq(seed):
    """
    test DeepQ on the uav_env(cartesian,discrete) 
    """
    # logger.configure()
    # set_global_seeds(SEED)
    # env = make_atari(ENV_ID)
    # env = bench.Monitor(env, logger.get_dir())
    # env = wrap_atari_dqn(env)
    """
    DQN(policy, env, gamma=0.99, learning_rate=0.0005, buffer_size=50000,
     exploration_fraction=0.1, exploration_final_eps=0.02, train_freq=1,
     batch_size=32, checkpoint_freq=10000, checkpoint_path=None,
     learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,
     prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4,
     prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06, 
     param_noise=False, verbose=0, tensorboard_log=None, _init_setup_model=True)

    """
    algo = 'DQN'
    num_timesteps = 1000000

    env = set_up_env(seed)

    global best_mean_reward, n_steps
    best_mean_reward, n_steps = -np.inf, 0

    model = DQN(env=env, policy=DQNMlpPolicy, learning_rate=1e-4, buffer_size=10000,
                exploration_fraction=0.1, exploration_final_eps=0.01, train_freq=4,
                learning_starts=10000, target_network_update_freq=1000,
                gamma=0.99, prioritized_replay=True, prioritized_replay_alpha=0.6,
                checkpoint_freq=10000, verbose=0,
                tensorboard_log="./logs/{}/tensorboard/{}/".format(EXPERIMENT_NATURE, algo))

    model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed,
                log_interval=500, tb_log_name="seed_{}".format(seed))

    model = DQN.load(log_dir + 'best_model.pkl')

    evaluation = evaluate_model(env, model, 100)
    os.makedirs('./logs/{}/csv/{}/'.format(EXPERIMENT_NATURE, algo), exist_ok=True)
    os.rename('/tmp/gym/monitor.csv', "./logs/{}/csv/{}/seed_{}.csv".format(EXPERIMENT_NATURE, algo, seed))
    env.close()
    del model, env
    gc.collect()
    return evaluation


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


# parametrize("policy", ['cnn', 'lstm'])
def train_acer(seed):
    """
    test ACER on the uav_env(cartesian,discrete)
    :param seed: random seed
    :return: evaluation
    """
    """
    ACER(policy, env, gamma=0.99, n_steps=20, num_procs=1, q_coef=0.5, ent_coef=0.01,
    max_grad_norm=10, learning_rate=0.0007, lr_schedule='linear', rprop_alpha=0.99,
    rprop_epsilon=1e-05, buffer_size=5000, replay_ratio=4, replay_start=1000, 
    correction_term=10.0, trust_region=True, alpha=0.99, delta=1, verbose=0, 
    tensorboard_log=None, _init_setup_model=True)
    """
    algo = 'ACER'
    num_timesteps = 3000000

    env = set_up_env(seed)

    global best_mean_reward, n_steps
    best_mean_reward, n_steps = -np.inf, 0

    model = ACER(policy=MlpPolicy, env=env, gamma=0.99, n_steps=20, num_procs=1,
                 q_coef=0.5, ent_coef=0.01, max_grad_norm=10, learning_rate=0.0007,
                 lr_schedule='linear', rprop_alpha=0.99, rprop_epsilon=1e-05,
                 buffer_size=5000, replay_ratio=4, replay_start=1000,
                 correction_term=10.0, trust_region=True, alpha=0.99, delta=1,
                 verbose=0, tensorboard_log="./logs/{}/tensorboard/{}/".format(EXPERIMENT_NATURE, algo))

    model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed,
                log_interval=500, tb_log_name="seed_{}".format(seed))

    model = ACER.load(log_dir + 'best_model.pkl')

    evaluation = evaluate_model(env, model, 100)
    os.makedirs('./logs/{}/csv/{}/'.format(EXPERIMENT_NATURE, algo), exist_ok=True)
    os.rename('/tmp/gym/monitor.csv', "./logs/{}/csv/{}/seed_{}.csv".format(EXPERIMENT_NATURE, algo, seed))
    env.close()
    del model, env
    gc.collect()
    return evaluation


def train_acktr(seed):
    """
    test ACKTR on the uav_env(cartesian,discrete) 
    """
    """
    ACKTR(policy, env, gamma=0.99, nprocs=1, n_steps=20, ent_coef=0.01, vf_coef=0.25, 
    vf_fisher_coef=1.0, learning_rate=0.25, max_grad_norm=0.5, kfac_clip=0.001, 
    lr_schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True, 
    async_eigen_decomp=False)
    """
    algo = 'ACKTR'
    num_timesteps = 3000000

    env = set_up_env(seed)

    global best_mean_reward, n_steps
    best_mean_reward, n_steps = -np.inf, 0

    model = ACKTR(policy=MlpPolicy, env=env, gamma=0.99, nprocs=1, n_steps=20,
                  ent_coef=0.01, vf_coef=0.25, vf_fisher_coef=1.0, learning_rate=0.25,
                  max_grad_norm=0.5, kfac_clip=0.001, lr_schedule='linear', verbose=0,
                  tensorboard_log="./logs/{}/tensorboard/{}/".format(EXPERIMENT_NATURE, algo),
                  _init_setup_model=True)
    # , async_eigen_decomp=False)

    model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed,
                log_interval=500, tb_log_name="seed_{}".format(seed))

    model = ACKTR.load(log_dir + 'best_model.pkl')

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

    model = TRPO.load(log_dir + 'best_model.pkl')

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

    model = PPO2.load(log_dir + 'best_model.pkl')

    evaluation = evaluate_model(env, model, 100)
    os.makedirs('./logs/{}/csv/{}/'.format(EXPERIMENT_NATURE, algo), exist_ok=True)
    os.rename('/tmp/gym/monitor.csv', "./logs/{}/csv/{}/seed_{}.csv".format(EXPERIMENT_NATURE, algo, seed))
    env.close()
    del model, env
    gc.collect()
    return evaluation


def train_model(name, parametrized_model, method, seed):
    """
    Generic call for training and evaluating a model

    :param name: method name e.g. 'DQN'
    :param parametrized_model: method call with all the parameters e.g. DQN(n_t=....)
    :param method: method used to call load from e.g. DQN
    :param seed: random seed
    :return: the evaluation from evaluate_model
    """

    """
    PPO2(policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=0.00025, 
    vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, 
    cliprange=0.2, verbose=0, tensorboard_log=None, _init_setup_model=True)
    """
    algo = name
    num_timesteps = 3000000

    env = set_up_env(seed)

    global best_mean_reward, n_steps
    best_mean_reward, n_steps = -np.inf, 0

    # Tested with n_steps=128
    model = parametrized_model

    model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed,
                log_interval=500, tb_log_name="seed_{}".format(seed))

    model = method.load(log_dir + 'best_model.pkl')

    evaluation = evaluate_model(env, model, 100)
    os.makedirs('./logs/{}/csv/{}/'.format(EXPERIMENT_NATURE, algo), exist_ok=True)
    os.rename('/tmp/gym/monitor.csv', "./logs/{}/csv/{}/seed_{}.csv".format(EXPERIMENT_NATURE, algo, seed))
    env.close()
    del model, env
    gc.collect()
    return evaluation
