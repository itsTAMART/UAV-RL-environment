from uav_enviroment.UAV_Environment import UAVEnv


class Curriculum_UAVEnv(UAVEnv):
    env = None

    curriculum = [
        {'n_obstacles': 0, 'threshold_distance': 40},
        {'n_obstacles': 0, 'threshold_distance': 20},
        {'n_obstacles': 1, 'threshold_distance': 40},
        {'n_obstacles': 2, 'threshold_distance': 40},
        {'n_obstacles': 3, 'threshold_distance': 40},
        {'n_obstacles': 4, 'threshold_distance': 40},
        {'n_obstacles': 5, 'threshold_distance': 40},
        {'n_obstacles': 6, 'threshold_distance': 40},
        {'n_obstacles': 7, 'threshold_distance': 40},
        {'n_obstacles': 8, 'threshold_distance': 40},
        {}

    ]

    def __init__(self, continuous=True, angular_movement=True, observation_with_image=False,
                 reset_always=True, controlled_speed=True):
        super().__init__(continuous=continuous, angular_movement=angular_movement,
                         observation_with_image=observation_with_image, reset_always=reset_always,
                         controlled_speed=controlled_speed)

    def reset(self):
        pass  # TODO

    def next_episode(self):
        """ Returns True if there have been more than 50% of successful episodes without crashes"""
        return (self.n_done - self.crashes) > (self.logging_episodes / 2)
