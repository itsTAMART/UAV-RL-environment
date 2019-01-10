from utils.launcher import *
from utils.evaluate_model import *
from experiments_continuous_cartesian import *

print("Starting experiments with cartesian dynamics and continuous action space \n \n")
results = run_experiments([
    # train_ddpg,
                           train_a2c,
                           train_a2c_recurrent,
                           train_trpo,
                           train_ppo2,
                           train_ppo2_recurrent
                           ])

print(results[results.columns[:4]])
