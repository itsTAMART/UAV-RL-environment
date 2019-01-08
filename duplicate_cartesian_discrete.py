from utils.launcher import *
from utils.evaluate_model import *
from experiments import *

print("Starting experiments with cartesian dynamics and discrete action space \n \n")
results = run_experiments([
    train_trpo,
    train_ppo2
])

print(results[results.columns[:4]])

plot_experiment(experiment='UAVenv_cartesian_discrete')
