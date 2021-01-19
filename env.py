from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.utils import seeding

import numpy as np
import math

class CustomMountainCarEnv(Continuous_MountainCarEnv):
    # By default,
    # self.max_speed = 0.07
    # self.power = 0.0015
    def __init__(self):
        super().__init__()
        self.time_limit = 400

    def reset(self, max_speed_current=None, power_current=None):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        if max_speed_current: self.max_speed = max_speed_current
        if power_current: self.power = power_current
        self.step_count = 0
        self.terminal_signal = False
        return np.array(self.state)

    def step(self, action, exclude_energy=False):

        self.step_count += 1
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0

        # Convert a possible numpy bool to a Python bool.
        self.terminal_signal = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        done = self.terminal_signal

        reward = 0
        if done:
            reward = 100.0

        if not exclude_energy:
            reward -= math.pow(action[0], 2) * 0.1
        else:
            reward -= 0.05

        self.state = np.array([position, velocity])
        done = done or (self.step_count == 1000)
        return self.state, reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed) # Action space is not seeded by default
        return [seed]
