from utils.launcher import *
from experiments.experiments_discrete_cartesian import *

print("Starting experiments with cartesian dynamics and discrete action space \n \n")
results = run_experiments([train_deepq,
                           train_a2c,
                           train_acer,
                           train_acktr,
                           train_trpo,
                           train_ppo2
                           ])

print(results[results.columns[:4]])
