import imageio
import cv2

from uav_enviroment.UAV_Environment import UAVEnv
from utils.logs_callback import *

from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy

NUM_CPU = 2
EXPERIMENT_NATURE = 'UAVenv_discrete_cartesian'
seed = 16


def setup_env_cart_discrete(seed):
    """
    Sets up the environment with cartesian observations and discrete action space.
    :param seed: random seed
    :return: the environment
    """
    env = UAVEnv(continuous=False, angular_movement=False, observation_with_image=False, reset_always=True)
    env.setup(n_obstacles=3, reset_always=True, threshold_dist=20)
    env.seed(seed)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    return env


set_up_env = setup_env_cart_discrete

algo = 'A2C'
num_timesteps = 3000000

env = set_up_env(seed)

global best_mean_reward, n_steps
best_mean_reward, n_steps = -np.inf, 0

# model = A2C(policy=MlpPolicy, env=env, gamma=0.99, n_steps=5, vf_coef=0.25,
#             ent_coef=0.01, max_grad_norm=0.5, learning_rate=0.0007, alpha=0.99,
#             epsilon=1e-05, lr_schedule='linear', verbose=1)
#
# model.learn(total_timesteps=num_timesteps, callback=callback, seed=seed,
#             log_interval=500)

model = A2C.load('/home/daniel/Desktop/demo/gym/best_model.pkl')
model.set_env(env)

images = []
obs = model.env.reset()
img = model.env.render(mode='rgb_array')

for i in range(3000):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _, _ = model.env.step(action)
    #     img = model.env.render(mode='rgb_array')
    #     cv2.imshow('image', img)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    model.env.render(mode='human')

imageio.mimsave('uav_learning.gif', [img for i, img in enumerate(images) if i % 5 == 0], fps=60)
