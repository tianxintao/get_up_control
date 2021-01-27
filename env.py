from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gym.utils import seeding
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.robot_locomotors import WalkerBase

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

class HumanoidStandup(WalkerBase):
    self_collision = True
    foot_list = ["right_foot", "left_foot","left_lower_arm", "right_lower_arm"]  #

    def __init__(self):
        WalkerBase.__init__(self,
                            'humanoid_symmetric.xml',
                            'torso',
                            action_dim=17,
                            obs_dim=43,
                            power=0.4)
        self.model_xml = "humanoid_standup.xml"
        # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25

    def reset(self, bullet_client):
        self._p = bullet_client
        self._p.setAdditionalSearchPath('./')
        WalkerBase.reset(self,bullet_client)

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power += [25, 25, 25]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [25, 25, 25]
        self.motors = [self.jdict[n] for n in self.motor_names]

        self.robot_body.reset_position([0, 0, 0.15])
        self.robot_body.reset_orientation([0, 0, 0, 1])
        self.initial_z = 0.15

    random_yaw = False
    random_lean = False

    def apply_action(self, a):
        assert (np.isfinite(a).all())
        for i, m, power in zip(range(17), self.motors, self.motor_power):
            m.set_motor_torque(float(power * self.power * np.clip(a[i], -1, +1)))

    def alive_bonus(self, z, pitch):
        return +1

    def calc_potential(self):
        return 0

    def calc_state(self):
        j = np.array([j.current_relative_position() for j in self.ordered_joints],
                     dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]

        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2]
                         )  # torso z is more informative than mean z
        self.body_real_xyz = body_pose.xyz()
        quat = self.robot_body.current_orientation()
        z = self.body_xyz[2]
        if self.initial_z == None:
            self.initial_z = z

        more = np.array([z,quat[0],quat[1],quat[2],quat[3]],dtype=np.float32)
        return np.clip(np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)


class HumanoidBulletEnv(WalkerBaseBulletEnv):

    def __init__(self, robot=None, render=False):
        if robot is None:
            self.robot = HumanoidStandup()
        else:
            self.robot = robot
        WalkerBaseBulletEnv.__init__(self, self.robot, render)


    def step(self, a):
        self._step_count += 1
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        done = self._step_count >= 1000
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        # potential_old = self.potential
        # self.potential = self.robot.calc_potential()
        # progress = float(self.potential - potential_old)

        for i, f in enumerate(
                self.robot.feet
        ):  # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        # electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean(
        # ))  # let's assume we have DC motor with controller, and reverse current braking
        # electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        #
        # joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        self.rewards = [
            self.robot.body_xyz[2]
        ]
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)


    def reset(self):
        WalkerBaseBulletEnv.reset(self)
        self._step_count = 0