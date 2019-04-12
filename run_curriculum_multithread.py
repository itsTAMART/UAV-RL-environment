import os
import datetime

if __name__ == '__main__':
    print('Startirng Curriculum Learning Experiments: \n'
          '     10 seeds with curriculum Learning \n'
          '     10 seeds without it')

    print('Creating the Experiment Log folder')
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = './logs/Experiment_CurriculumLearning_{}/'.format(time)
    os.makedirs(experiment_dir, exist_ok=True)

    pass
