import os
import datetime

# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

from uav_enviroment.UAV_Environment import UAVEnv

from utils.evaluate_model import *

# if __name__ == '__main__':

SEED = 16

# Create and wrap the environment
env = UAVEnv(continuous=False, angular_movement=False, observation_with_image=False, reset_always=True)
env.setup(n_obstacles=2, reset_always=True, threshold_dist=20)
env.seed(SEED)
env = DummyVecEnv([lambda: env])

model = DQN(MlpPolicy, env,  verbose=1)
# model.load(load_path, env=env)
model = model.learn(total_timesteps=5000, log_interval=50)

results = evaluate_model(env=env, model=model,n_episodes=50, render_eval=False)

assert results is not None
print(results)
