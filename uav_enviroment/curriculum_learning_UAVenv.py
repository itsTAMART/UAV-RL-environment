from uav_enviroment.UAV_Environment import UAVEnv


class Curriculum_UAVEnv(UAVEnv):

    curriculum = [
        {'n_obstacles': 0, 'threshold_distance': 60},
        {'n_obstacles': 0, 'threshold_distance': 40},
        {'n_obstacles': 0, 'threshold_distance': 20},
        {'n_obstacles': 1, 'threshold_distance': 40},
        {'n_obstacles': 2, 'threshold_distance': 40},
        {'n_obstacles': 3, 'threshold_distance': 40},
        {'n_obstacles': 4, 'threshold_distance': 40},
        {'n_obstacles': 5, 'threshold_distance': 40},
        {'n_obstacles': 6, 'threshold_distance': 40},
        {'n_obstacles': 7, 'threshold_distance': 40},
        {'n_obstacles': 8, 'threshold_distance': 40}

    ]

    difficulty_level = 0

    # def __init__(self):
    #     super().__init__()

    def reset(self):
        if self.is_next_difficulty():
            self.difficulty_level += 1
            level = self.curriculum[self.difficulty_level] \
                if self.difficulty_level < len(self.curriculum) \
                else self.curriculum[-1]

            self.setup(n_obstacles=level['n_obstacles'], threshold_dist=level['threshold_distance'],
                       reset_always=True, reward_sparsity=True)
            print('new level: {}'.format(level))

        return super().reset()

    def is_next_difficulty(self):
        """ Returns True if there have been more than 50% of successful episodes without crashes"""
        return (self.n_done - (self.crashes+ self.oob)) > (self.logging_episodes / 2)

