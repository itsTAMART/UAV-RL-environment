import os
import datetime

# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

from uav_enviroment.UAV_Environment import UAVEnv

if __name__ == '__main__':
    # Create log dir
    log_dir = "/tmp/uav_env/"
    load_path = './models/DQN_disc_cartesian/2018_11_05_16:58.pkl'
    experiment_name = 'DQN_disc_cartesian'
    models_dir = './models/'+ experiment_name+'/'
    t = datetime.datetime.now()
    date = t.strftime('%Y_%m_%d_%H:%M')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)


    # Create and wrap the environment
    env = UAVEnv(continuous=False, angular_movement=False, observation_with_image=False, reset_always=True)
    env.setup(n_obstacles=2, reset_always=True, threshold_dist=20)
    env = DummyVecEnv([lambda: env])

    model = DQN(MlpPolicy, env,  verbose=1, tensorboard_log=log_dir)
    # model.load(load_path, env=env)
    model = model.learn(total_timesteps=150000, log_interval=50, tb_log_name=experiment_name+date)
    model.save(models_dir + date)



