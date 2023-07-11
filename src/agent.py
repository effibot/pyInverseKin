import re
from sqlite3 import adapt

from matplotlib import markers
import constants as c
import numpy as np
from numpy import ndarray, sin, cos
from numpy.linalg import norm
from tiles3 import IHT, tiles

np.random.seed(42)


class agent(object):
    def __init__(self, target):
        self.constants = {"l1": c.L1, "l2": c.L2, "m1": c.M1, "m2": c.M2, "g": c.g}
        self.is_sat = True
        self.state = np.zeros((1, 4))
        self.time_step = c.TIME_STEP
        self.actions: np.ndarray = [1e5, 1e4] * np.asarray(
            (
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 0],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            )
        )
        self.actions_history = np.zeros((c.NUM_EPISODES, c.MAX_STEPS, 2))
        self.reward_history = np.zeros((c.NUM_EPISODES, c.MAX_STEPS))
        self.position_history = np.zeros((c.NUM_EPISODES, c.MAX_STEPS, 2))
        self.target: ndarray = target

    def h_q(self, q1, q2):
        x = c.L1 * cos(q1) + c.L2 * cos(q1 + q2)
        y = c.L1 * sin(q1) + c.L2 * sin(q1 + q2)
        return np.asarray([x, y])

    def get_p(self):
        return self.h_q(self.state[0], self.state[1])

    def get_state(self):
        return self.state

    def get_action(self, index):
        return self.actions[index]

    def update_pos_history(self, episode, step):
        self.position_history[episode, step, :] = self.get_state()[0:2]

    def update_actions_history(self, episode, step, action):
        self.actions_history[episode, step] = action

    def update_reward_history(self, episode, step, reward):
        self.reward_history[episode, step] += reward

    def get_histories(self):
        return {
            "position": self.position_history,
            "actions": self.actions_history,
            "reward": self.reward_history,
        }

    def reset(self):
        ws = self.generate_ws_curves()
        i, j = np.random.randint(0, ws[0].shape[0]), np.random.randint(
            0, ws[1].shape[0]
        )
        # randomize initial joint angles
        self.state = np.asarray(
            [
                ws[0][i],  # q1
                ws[1][j],  # q2
                # velocities are set to zero when resetting
                0,
                0,
            ]
        )

    def generate_ws_curves(self, n=1000):
        """
        This function generates workspace curves by computing the forward kinematics for a grid of
        points within a defined range and filtering out points outside the workspace.

        :param n: The number of points to use to draw the workspace curves. The higher the number, the
        smoother the curve
        :return: two arrays, X and Y, which contain the coordinates of the points that define the
        workspace curves.
        """
        # define how many points to use to draw the curves. The higher the number, the smoother the curve
        n_points = n
        # define the range of angles to use
        t1 = np.linspace(c.WORKING_RANGE[0, 0], c.WORKING_RANGE[0, 1], n_points)[:-1]
        t2 = np.linspace(c.WORKING_RANGE[1, 0], c.WORKING_RANGE[1, 1], n_points)[:-1]
        # create the grid of points
        grid = np.meshgrid(t1, t2)
        # compute the forward kinematics for each point in the grid
        X, Y = self.h_q(grid[0], grid[1])
        # filter out the points that are outside the workspace
        X, Y = X[self.is_inside_ws(X, Y)], Y[self.is_inside_ws(X, Y)]
        # filter out duplicate of pairs of points
        X, Y = np.unique([X, Y], axis=1)
        # return the points
        return X, Y

    def is_inside_ws(self, x, y):
        """Check if the current state is inside the workspace of the robot."""
        cond = ((x**2 + y**2) < (c.L1 + c.L2) ** 2) & (
            (x**2 + y**2) > (c.L1 - c.L2) ** 2
        )
        return cond

    def compute_distance(self, state):
        """Compute the distance between the end effector and the target."""
        p = self.h_q(state[0], state[1])
        return norm(p - self.target)

    def get_reward(self, state, episode):
        """
        Returns the reward for the current state.
        """
        # compute the distance between the end effector and the target
        covered_distance = self.compute_distance(state)
        # compute the reward
        reward = -(covered_distance**2)
        # return the reward
        return reward

    def is_done(self):
        """
        Checks if the current state is terminal.
        """
        p = self.h_q(self.state[0], self.state[1])
        if norm(p - self.target) <= c.EPSILON:
            return True
        return False

    # define the vector field function
    def dxdt(self, x, tau):
        """This function computes the time derivative of the state vector x.

        Args:
            x (np.array): state vector
            constant (dictionary): constants dictionary for the system
            tau (list): torques

        Returns:
            np.array: time derivative of the state vector
        """
        # unpack the state vector
        x1, x2, x3, x4 = x
        # unpack the constants
        l1 = self.constants["l1"]
        l2 = self.constants["l2"]
        m1 = self.constants["m1"]
        m2 = self.constants["m2"]
        g = self.constants["g"]
        # unpack the torques
        t1, t2 = tau
        # compute the time derivative of the state vector
        dx1dt = x3
        dx2dt = x4
        dx3dt = (
            l2 * t1
            - l2 * t2
            - l2 * x3
            + l2 * x4
            - l1 * t2 * cos(x2)
            + l1 * x4 * cos(x2)
            + l1 * l2**2 * m2 * x3**2 * sin(x2)
            + l1 * l2**2 * m2 * x4**2 * sin(x2)
            + g * l1 * l2 * m1 * cos(x1)
            + g * l1 * l2 * m2 * cos(x1)
            - g * l1 * l2 * m2 * cos(x1 + x2) * cos(x2)
            + l1**2 * l2 * m2 * x3**2 * cos(x2) * sin(x2)
            + 2 * l1 * l2**2 * m2 * x3 * x4 * sin(x2)
        ) / (l1**2 * l2 * m1 + l1**2 * l2 * m2 - l1**2 * l2 * m2 * cos(x2) ** 2)
        dx4dt = -(
            l2**2 * m2 * t1
            - l1**2 * m2 * t2
            - l1**2 * m1 * t2
            - l2**2 * m2 * t2
            + l1**2 * m1 * x4
            + l1**2 * m2 * x4
            - l2**2 * m2 * x3
            + l2**2 * m2 * x4
            + l1 * l2**3 * m2**2 * x3**2 * sin(x2)
            + l1**3 * l2 * m2**2 * x3**2 * sin(x2)
            + l1 * l2**3 * m2**2 * x4**2 * sin(x2)
            - g * l1**2 * l2 * m2**2 * cos(x1 + x2)
            + g * l1 * l2**2 * m2**2 * cos(x1)
            + l1 * l2 * m2 * t1 * cos(x2)
            - 2 * l1 * l2 * m2 * t2 * cos(x2)
            - l1 * l2 * m2 * x3 * cos(x2)
            + 2 * l1 * l2 * m2 * x4 * cos(x2)
            + l1**3 * l2 * m1 * m2 * x3**2 * sin(x2)
            + 2 * l1 * l2**3 * m2**2 * x3 * x4 * sin(x2)
            - g * l1 * l2**2 * m2**2 * cos(x1 + x2) * cos(x2)
            + 2 * l1**2 * l2**2 * m2**2 * x3**2 * cos(x2) * sin(x2)
            + l1**2 * l2**2 * m2**2 * x4**2 * cos(x2) * sin(x2)
            + g * l1**2 * l2 * m2**2 * cos(x1) * cos(x2)
            - g * l1**2 * l2 * m1 * m2 * cos(x1 + x2)
            + g * l1 * l2**2 * m1 * m2 * cos(x1)
            + 2 * l1**2 * l2**2 * m2**2 * x3 * x4 * cos(x2) * sin(x2)
            + g * l1**2 * l2 * m1 * m2 * cos(x1) * cos(x2)
        ) / (
            -(l1**2) * l2**2 * m2**2 * cos(x2) ** 2
            + l1**2 * l2**2 * m2**2
            + m1 * l1**2 * l2**2 * m2
        )

        return np.asarray([dx1dt, dx2dt, dx3dt, dx4dt])

    # define the vector field function without friction
    def dxdt_no_friction(self, x, tau):
        """This function computes the time derivative of the state vector x without friction.

        Args:
            x (np.array): state vector
            constant (dictionary): constants dictionary for the system
            tau (list): torques

        Returns:
            np.array: time derivative of the state vector
        """
        # unpack the state vector
        x1, x2, x3, x4 = x[0]
        # unpack the constants
        l1 = self.constants["l1"]
        l2 = self.constants["l2"]
        m1 = self.constants["m1"]
        m2 = self.constants["m2"]
        g = self.constants["g"]
        # unpack the torques
        t1, t2 = tau
        # compute the time derivative of the state vector
        dx1dt = x3
        dx2dt = x4
        dx3dt = (
            2 * l2 * t1
            - 2 * l2 * t2
            - 2 * l1 * t2 * cos(x2)
            - g * l1 * l2 * m2 * cos(x1 + 2 * x2)
            + l1**2 * l2 * m2 * x3**2 * sin(2 * x2)
            + 2 * l1 * l2**2 * m2 * x3**2 * sin(x2)
            + 2 * l1 * l2**2 * m2 * x4**2 * sin(x2)
            + 2 * g * l1 * l2 * m1 * cos(x1)
            + g * l1 * l2 * m2 * cos(x1)
            + 4 * l1 * l2**2 * m2 * x3 * x4 * sin(x2)
        ) / (l1**2 * l2 * (2 * m1 + m2 - m2 * cos(2 * x2)))
        dx4dt = -(
            l2**2 * m2 * t1
            - l1**2 * m2 * t2
            - l1**2 * m1 * t2
            - l2**2 * m2 * t2
            + l1 * l2**3 * m2**2 * x3**2 * sin(x2)
            + l1**3 * l2 * m2**2 * x3**2 * sin(x2)
            + l1 * l2**3 * m2**2 * x4**2 * sin(x2)
            + g * l1 * l2**2 * m2**2 * cos(x1)
            + l1**2 * l2**2 * m2**2 * x3**2 * sin(2 * x2)
            + (l1**2 * l2**2 * m2**2 * x4**2 * sin(2 * x2)) / 2
            + l1 * l2 * m2 * t1 * cos(x2)
            - 2 * l1 * l2 * m2 * t2 * cos(x2)
            + l1**3 * l2 * m1 * m2 * x3**2 * sin(x2)
            + 2 * l1 * l2**3 * m2**2 * x3 * x4 * sin(x2)
            + g * l1**2 * l2 * m2**2 * sin(x1) * sin(x2)
            + g * l1 * l2**2 * m1 * m2 * cos(x1)
            - g * l1 * l2**2 * m2**2 * cos(x1) * cos(x2) ** 2
            + l1**2 * l2**2 * m2**2 * x3 * x4 * sin(2 * x2)
            + g * l1 * l2**2 * m2**2 * cos(x2) * sin(x1) * sin(x2)
            + g * l1**2 * l2 * m1 * m2 * sin(x1) * sin(x2)
        ) / (l1**2 * l2**2 * m2 * (-m2 * cos(x2) ** 2 + m1 + m2))

        return np.asarray([dx1dt, dx2dt, dx3dt, dx4dt])

    def step(self, x, tau):
        """This function computes the next state of the system with RK4 method.

        Args:
            x (np.array): state vector
            tau (list): torques

        Returns:
            np.array: next state of the system
        """
        # RK4 integration
        k1 = self.dxdt(x, tau)
        k2 = self.dxdt(x + self.time_step * k1 / 2, tau)
        k3 = self.dxdt(x + self.time_step * k2 / 2, tau)
        k4 = self.dxdt(x + self.time_step * k3, tau)
        # update the state
        self.state += self.time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # saturate
        if self.is_sat:
            self.state[0] = np.clip(
                self.state[0], c.WORKING_RANGE[0, 0], c.WORKING_RANGE[0, 1]
            )
            self.state[1] = np.clip(
                self.state[1], c.WORKING_RANGE[1, 0], c.WORKING_RANGE[1, 1]
            )
        return self.get_state()

    def step_no_friction(self, x, tau):
        """This function computes the next state of the system with RK4 method without friction.

        Args:
            x (np.array): state vector
            tau (list): torques

        Returns:
            np.array: next state of the system
        """
        # RK4 integration
        k1 = self.dxdt_no_friction(x, tau)
        k2 = self.dxdt_no_friction(x + self.time_step * k1 / 2, tau)
        k3 = self.dxdt_no_friction(x + self.time_step * k2 / 2, tau)
        k4 = self.dxdt_no_friction(x + self.time_step * k3, tau)
        # update the state
        self.state += self.time_step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # saturate
        if self.is_sat:
            self.state[0, 0] = np.clip(
                self.state[0, 0], c.WORKING_RANGE[0, 0], c.WORKING_RANGE[0, 1]
            )
            self.state[0, 1] = np.clip(
                self.state[0, 1], c.WORKING_RANGE[1, 0], c.WORKING_RANGE[1, 1]
            )
        return self.get_state()

    def get_state_reward_transition(self, state, action, episode):
        """This function computes the next state and the reward for the current state and action.

        Args:
            state (np.array): current state
            action (np.array): current action

        Returns:
            np.array: next state
            float: reward
        """
        # compute the next state
        next_state = self.step(state, action)
        # compute the reward
        reward = self.get_reward(state, episode)
        return next_state, reward

    def get_state_reward_transition_no_friction(self, state, action, episode):
        """This function computes the next state and the reward for the current state and action.

        Args:
            state (np.array): current state
            action (np.array): current action

        Returns:
            np.array: next state
            float: reward
        """
        # compute the next state
        next_state = self.step_no_friction(state, action)
        # compute the reward
        reward = self.get_reward(state, episode)
        return next_state, reward

    def update_histories(self, episode, step, action, reward):
        # update the position history for the current episode at the current step
        self.update_pos_history(episode, step)
        # update the reward history for the current episode until the current step
        self.update_reward_history(episode, step, reward)
        # update the actions history for the current episode at the current step
        self.update_actions_history(episode, step, action)


class ws_env:
    def __init__(self) -> None:
        self.iht = IHT(c.NUM_FEATURES)
        self.W = np.zeros((c.NUM_ACTIONS, c.NUM_FEATURES))
        self.actions: np.ndarray = [1e5, 1e4] * np.asarray(
            [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 0],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            ]
        )

    def get_active_tiles(self, state, action):
        # get the indices of the active tiles

        # extract the state variables
        q1 = state[0]
        q2 = state[1]
        q1_d = state[2]
        q2_d = state[3]
        # define the scaling factor
        q1_sf = c.NUM_TILINGS * q1 / (c.WORKING_RANGE[0, 1] - c.WORKING_RANGE[0, 0])
        q2_sf = c.NUM_TILINGS * q2 / (c.WORKING_RANGE[1, 1] - c.WORKING_RANGE[1, 0])
        q1_d_sf = (
            c.NUM_TILINGS
            * q1_d
            / (c.WORKING_VELOCITIES[0, 1] - c.WORKING_VELOCITIES[0, 0])
        )
        q2_d_sf = (
            c.NUM_TILINGS
            * q2_d
            / (c.WORKING_VELOCITIES[1, 1] - c.WORKING_VELOCITIES[1, 0])
        )
        # get the indices of the active tiles
        active_tiles = tiles(
            self.iht, c.NUM_TILINGS, [q1_sf, q2_sf, q1_d_sf, q2_d_sf], action
        )
        return active_tiles

    def epsGreedy(self, state, init=True):
        if np.random.rand() < c.EPSILON or init:
            return self.actions[np.random.randint(0, c.NUM_ACTIONS)]
        else:
            q_values = np.zeros((c.NUM_ACTIONS, 1))
            for i in range(c.NUM_ACTIONS):
                # get the active tiles
                active_tiles = self.get_active_tiles(state, self.actions[i])
                # compute the value of the current state-action pair
                q_values[i] = np.sum(self.W[i, active_tiles])
        return self.actions[np.argmax(q_values)]

    def step_update(self, state, action, reward, next_state, next_action):
        # get active tiles
        active_tiles = self.get_active_tiles(state, action)
        # compute the value of the current state-action pair
        a_i = np.where((self.actions == action).all(axis=1))[0][0]
        q_value = np.sum(self.W[a_i, active_tiles])
        # get active tiles for the next state
        next_active_tiles = self.get_active_tiles(next_state, next_action)
        # compute the value of the next state-action pair
        a_i_next = np.where((self.actions == next_action).all(axis=1))[0][0]
        next_q_value = np.sum(self.W[a_i_next, next_active_tiles])
        # update the weights
        delta = reward + c.GAMMA * next_q_value - q_value
        self.W[a_i_next, next_active_tiles] += c.ALPHA * delta

    def episode_loop(self, agent, episode):
        # reset the agent
        agent.reset()
        # reset the environment
        self.reset()
        # get the initial state
        state = agent.get_state()
        # get the initial action
        action = self.epsGreedy(state)
        # loop until the episode is done
        for step in range(c.MAX_STEPS):
            # get the next state and reward
            next_state, reward = agent.get_state_reward_transition(
                state, action, episode
            )
            # get the next action
            next_action = self.epsGreedy(next_state, False)
            # update the weights
            self.step_update(state, action, reward, next_state, next_action)
            # update agent's histories
            agent.update_histories(episode, step, action, reward)
            # checks if the episode is done
            if agent.is_done():
                break
            # update the state and action
            state = next_state
            action = next_action

    def reset(self):
        self.iht = IHT(c.NUM_FEATURES)
        #self.W = np.zeros((c.NUM_ACTIONS, c.NUM_FEATURES))


import matplotlib.pyplot as plt

if __name__ == "__main__":
    # initialize target and agent
    target = np.asarray([c.L1 / 3 * 2 + c.L2, 50])
    arm = agent(target)
    env = ws_env()
    fig, ax = plt.subplots(1, 2, figsize=(3, 7))
    ax[0].set_title("visited q1-q2 pairs")
    ax[0].legend()
    ax[1].set_title("target and end effector")
    ax[1].plot(target[0], target[1], "-r*")
    ax[1].legend()
    for episode in range(3):
        print(f"Episode {episode}")
        env.episode_loop(arm, episode)
        # print(arm.get_histories())
        # plot visited points
        pos = arm.get_histories()["position"][episode]
        ef = arm.h_q(pos[:, 0], pos[:, 1])
        ax[0].plot(pos[:, 0], pos[:, 1], "o")
        ax[1].plot(ef[0], ef[1], "o", markersize=1)
        #ax[1].legend(["target", f"end effector in {episode}"])
    plt.show()
