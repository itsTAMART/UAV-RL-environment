import os
import datetime
import os
import platform
from joblib import Parallel, delayed


def experiment(exptype, exp_dir, seed):
    if platform.system() == 'Windows':
        _ = os.system(
            'python experiments/experiments_curriculum_learning.py {0} {1}/{0}/ {2} > {1}/{0}/seed_{2}.txt'.format(
                exptype, exp_dir, seed))  # Windows order
    else:
        _ = os.system(
            'python3 experiments/experiments_curriculum_learning.py {0} {1}/{0}/ {2} > {1}/{0}/seed_{2}.txt'.format(
                exptype, exp_dir, seed))  # Linux order


if __name__ == '__main__':
    print('Startirng Curriculum Learning Experiments: \n'
          '     10 seeds with curriculum Learning \n'
          '     10 seeds without it')

    print('Creating the Experiment Log folder')
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = './logs/Experiment_CurriculumLearning_{}/'.format(time)
    os.makedirs(experiment_dir, exist_ok=True)

    exp_types = ['curriculum', 'standard']
    for exp_type in exp_types:
        os.makedirs(experiment_dir + '/{}/'.format(exp_type), exist_ok=True)

    n_seeds = 10

    # TRAINING LOOP!!!
    num_cores = 20
    _ = Parallel(n_jobs=num_cores, verbose=5) \
        (delayed(experiment)(exptype=exp_type, exp_dir=experiment_dir + '/{}/'.format(exp_type), seed=seed)
         for exp_type in exp_types for seed in range(n_seeds))
