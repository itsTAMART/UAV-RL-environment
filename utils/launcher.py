import gc
import time
import pandas as pd

SEEDS = [16, 8, 42, 458912811, 969535,
         7261, 3847, 618526, 100, 1]

result_names = ['name',
                'seed',
                'score',
                'execution time',
                'n_good',
                'n_bad',
                'n_timeout',
                'avg_good_ep_len',
                'avg_bad_ep_len']


def run_experiment(experiment, seed):
    """
    Runs the passed experiment function and returns the evaluation metrics
    :param experiment: function to run a experiment
    :param seed: random seed
    :return: the score and the evaluation metrics
    """
    # setup()
    # model =  preprocess(df)
    # model.train(callback)
    # score = (modified by callback)

    print('seed: {}'.format(seed))
    # TODO check the seed passing
    evaluation = experiment(seed)

    return evaluation


def run_experiments(experiments, seeds=SEEDS):
    print('Running {} experiments'.format(len(experiments)))
    _ = [print(experiment.__name__) for experiment in experiments]
    print('\n')
    results = []
    for experiment in experiments:
        print('#'*80)
        print('Running: {}'.format(experiment.__name__))

        for seed in seeds:
            print('-' * 80)
            start = time.time()
            evaluation = run_experiment(experiment, seed)
            execution_time = time.time() - start

            print('Took {}s'.format(round(execution_time, 2)))
            print('-' * 80)

            [score, n_good, n_timeout, n_bad, avg_good_ep_len, avg_bad_ep_len] = evaluation

            results.append({
                'name': experiment.__name__,
                'seed': seed,
                'score': score,
                'execution time': f'{round(execution_time, 2)}s',
                'n_good': n_good,
                'n_timeout': n_timeout,
                'n_bad': n_bad,
                'avg_good_ep_len': avg_good_ep_len,
                'avg_bad_ep_len': avg_bad_ep_len
            })

            gc.collect()

        print('#' * 80)
        print('\n')

    return pd.DataFrame(results, columns=result_names).sort_values(by='score', ascending=False)
