import constants as c
import numpy as np
from numpy import ndarray, sin, cos
from numpy.linalg import norm
from tiles3 import IHT, tiles

#! To test unseeded agent, comment the following line
np.random.seed(42)


class agent(object):
    def __init__(self, target, env=None):
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
        self.achieved_target = False
        self.ws = self.generate_ws_curves()
        self.env = env

    def h_q(self, q1, q2):
        """Compute the forward kinematics of the robot.

        Args:
            q1: The angle of the first joint. Rad.
            q2: The angle of the second joint. Rad.

        Returns:
            A tuple containing the x and y coordinates of the end effector.
        """
        x = c.L1 * cos(q1) + c.L2 * cos(q1 + q2)
        y = c.L1 * sin(q1) + c.L2 * sin(q1 + q2)
        return np.asarray([x, y])

    def get_p(self):
        """Compute the forward kinematics of the robot at the current state.

        Returns:
            ndarray: The x and y coordinates of the end effector.
        """
        return self.h_q(self.state[0], self.state[1])

    # Getter methods
    def get_state(self):
        return self.state

    def get_action(self, index):
        return self.actions[index]

    # update histories methods
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

    def update_histories(self, episode, step, action, reward):
        # update the position history for the current episode at the current step
        self.update_pos_history(episode, step)
        # update the reward history for the current episode until the current step
        self.update_reward_history(episode, step, reward)
        # update the actions history for the current episode at the current step
        self.update_actions_history(episode, step, action)

    # reset the agent state
    def reset(self):
        i, j = np.random.randint(0, self.ws[0].shape[0]), np.random.randint(0, self.ws[1].shape[0])
        # randomize initial joint angles
        self.state = np.asarray(
            [
                self.ws[0][i],  # q1
                self.ws[1][j],  # q2
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
        cond = ((x**2 + y**2) < (c.L1 + c.L2) ** 2) & ((x**2 + y**2) > (c.L1 - c.L2) ** 2)
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
        reward = covered_distance**2
        # get the last assigned reward index for the current episode
        last_index = np.nonzero(self.reward_history[episode])[0]
        if last_index.size > 0:
            if -np.log(reward) == self.reward_history[episode, last_index[-1]]:
                reward *= 1.1
        # return the reward
        return reward

    def is_done(self):
        """
        Checks if the current state is terminal.
        """
        if self.compute_distance(self.state) <= c.GAMMA:
            self.achieved_target = True
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
        ) / (-(l1**2) * l2**2 * m2**2 * cos(x2) ** 2 + l1**2 * l2**2 * m2**2 + m1 * l1**2 * l2**2 * m2)

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
            # self.state[0] = np.clip(self.state[0], c.WORKING_RANGE[0, 0], c.WORKING_RANGE[0, 1])
            self.state[1] = np.clip(self.state[1], c.WORKING_RANGE[1, 0], c.WORKING_RANGE[1, 1])
            self.state[2] = np.clip(self.state[2], c.WORKING_VELOCITIES[0, 0], c.WORKING_VELOCITIES[0, 1])
            self.state[3] = np.clip(self.state[3], c.WORKING_VELOCITIES[1, 0], c.WORKING_VELOCITIES[1, 1])
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

    def load_policy(self, W, iht):
        self.env.iht = iht
        self.env.W = W
        
    def play(self, env=None):
        # assign the environment just in case we are not loading the policy
        if env is not None:
            self.env = env
        # reset the agent state and the target reached flag
        self.state = c.init_conditions
        self.achieved_target = False
        # take the first action as [0,0]
        action = [0, 0]
        # loop until the target is reached or the maximum number of steps is reached
        while not self.achieved_target:
            # compute the active tiles
            active_tiles = self.env.get_active_tiles(self.get_state(), action)
            # compute the Q values for the current state
            Q = np.sum(self.env.W[:, active_tiles], axis=1)
            # select the action
            action = self.actions[np.argmax(Q)]
            # compute the next state and the reward
            next_state, reward = self.get_state_reward_transition(self.get_state(), action, 0)
            # update the histories
            self.update_histories(0, 0, action, reward)
            # update the state
            self.state = next_state
            # check if the target is reached
            if self.is_done():
                break
        

#! Run the following lines to test the agent

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Environment import environ as ws_env
    # initialize target and agent
    target = np.asarray([c.L1 / 3 * 2 + c.L2, 50])
    arm = agent(target)
    env = ws_env()
    workspace = arm.generate_ws_curves()
    # plot setup
    fig, ax = plt.subplots(1, 2, figsize=(3, 7))
    ax[0].set_title("visited q1-q2 pairs globally")
    ax[1].axis("equal")
    # start learning
    for episode in range(c.NUM_EPISODES):
        print(f"Episode {episode}")
        env.episode_loop(arm, episode)
        pos = arm.get_histories()["position"][episode]
        pos = pos[pos[:, 0] != 0]
        ax[0].plot(pos[:, 0], pos[:, 1], "o")
        ef = arm.h_q(pos[:, 0], pos[:, 1])
        if episode % 5 == 0:
            # clear the axes of the trajectory plot
            ax[1].cla()
            # plot the results
            ax[1].set_title(f"End-Effector trajectory at episode {episode}")
            ax[1].plot(workspace[0], workspace[1], "mo", markersize=0.01)
            ax[1].plot(target[0], target[1], "-r*")
            ax[1].plot(ef[0, 0], ef[1, 0], "go")
            ax[1].plot(ef[0], ef[1], "o", markersize=1)
            ax[1].legend(["workspace", "target", "starting point", "end effector"])
            plt.show(block=False)
            plt.pause(0.1)
        if arm.achieved_target:
            print(f"Target reached in episode {episode}")
            # clear the axes of the trajectory plot
            ax[1].cla()
            # plot the results
            ax[1].set_title(f"End-Effector reached target at episode {episode}")
            ax[1].plot(workspace[0], workspace[1], "mo", markersize=0.01)
            ax[1].plot(target[0], target[1], "-r*")
            ax[1].plot(ef[0, 0], ef[1, 0], "go")
            ax[1].plot(ef[0], ef[1], "o", markersize=1)
            ax[1].legend(["workspace", "target", "starting point", "end effector"])
            plt.show()
            break
