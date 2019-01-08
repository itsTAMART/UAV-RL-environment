from utils.launcher import *
from utils.evaluate_model import *

ENV_ID = 'BreakoutNoFrameskip-v4'
SEED = 3
NUM_TIMESTEPS = 500
NUM_CPU = 2



def mock_up_test(name):
    print('Training {}...'.format(name))
    n = np.random.randn()
    time.sleep((n ** 2) / 6)
    evaluation = [n, 1 * n, n ** 2, n + 3, 4 * n, n * .5]

    return evaluation

def test_a2c():
    return mock_up_test('a2c')

def test_acer():
    return mock_up_test('acer')

def test_acktr():
    return mock_up_test('acktr')

def test_deepq():
    return mock_up_test('deepq')

def test_ppo1():
    return mock_up_test('ppo1')

def test_ppo2():
    return mock_up_test('ppo2')

def test_trpo():
    return mock_up_test('trpo')



results = run_experiments([
    test_a2c,
    test_acer,
    test_acktr,
    test_deepq,
    test_trpo,
    test_ppo1,
    test_ppo2
])


print(results[results.columns[:4]])