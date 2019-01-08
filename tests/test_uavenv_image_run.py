import os
import datetime

# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
# from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
# from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, TRPO, DDPG, DQN

from UAV_Environment import UAVEnv

if __name__ == '__main__':
    # Create log dir
    log_dir = "/tmp/BIG_uav_env/"
    load_path = './models/DQN_img_nature/2018_11_07_13:33.pkl'
    experiment_name = 'DQN_img_nature'
    models_dir = './models/'+ experiment_name+'/'
    t = datetime.datetime.now()
    date = t.strftime('%Y_%m_%d_%H:%M')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)


    # Create and wrap the environment
    env = UAVEnv(continuous=False, angular_movement=False, observation_with_image=True, reset_always=True)
    env.setup(n_obstacles=2, reset_always=True, threshold_dist=20)
    env = DummyVecEnv([lambda: env])

    model = DQN(CnnPolicy, env,  verbose=1, tensorboard_log=log_dir)
    # model.load(load_path, env=env)

    # for i in range(10):
    model = model.learn(total_timesteps=300000, log_interval=5, tb_log_name=experiment_name+date)
    model.save(models_dir + date)


    env = UAVEnv(continuous=False, angular_movement=False, observation_with_image=True, reset_always=True)
    env.setup(map_size_x=500, map_size_y=500, n_obstacles=5, reset_always=True, threshold_dist=40, max_timestep=1500)
    env = DummyVecEnv([lambda: env])


    model.set_env(env)
    success = 0
    total_epi = 50
    for _ in range(total_epi):
        done = [False]
        obs = env.reset()
        timestep_render = 0
        while not done[0]:
            timestep_render += 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if timestep_render % 5 == 0 or done[0]:
                env.render()

        if rewards[0] > 0 :
            success +=1

    print(experiment_name)
    print('{}/{} successful episodes'.format(success,total_epi))



