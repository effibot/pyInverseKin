from tkinter import W
from requests import get
import constants as c
import numpy as np
from tiles3 import IHT, tiles
from agent2 import agent
from numpy.linalg import norm
import pickle


class environ(object):
    def __init__(self, agent=None, exploring_start=False):
        self.agent: agent = agent
        self.target: np.ndarray = c.TARGET
        self.iht = IHT(c.NUM_FEATURES)
        self.exploring_start = exploring_start
        # self.W = np.zeros((c.NUM_ACTIONS, c.NUM_FEATURES))
        self.W = np.random.rand(c.NUM_ACTIONS, c.NUM_FEATURES)
        self.reward_scale = 1
        self.actions_history = np.zeros((c.NUM_EPISODES, c.MAX_STEPS + 1, 2))
        self.reward_history = np.zeros((c.NUM_EPISODES, c.MAX_STEPS + 1))
        self.position_history = np.zeros((c.NUM_EPISODES, c.MAX_STEPS + 1, 2))
        self.vel_history = np.zeros((c.NUM_EPISODES, c.MAX_STEPS + 1, 2))
        self.cost_history = np.zeros((c.NUM_EPISODES, 1))

    def get_agent(self):
        return self.agent

    def get_r_scale(self):
        return self.reward_scale

    def set_r_scale(self, r_scale):
        self.reward_scale = r_scale

    def set_iht(self, iht):
        self.iht = iht

    def set_pos_history(self, episode, step, pos):
        self.position_history[episode, step] = pos

    def set_vel_history(self, episode, step, vels):
        self.vel_history[episode, step] = vels

    def set_action_history(self, episode, step, action):
        self.actions_history[episode, step] = action

    def set_reward_history(self, episode, step, reward):
        self.reward_history[episode, step] = reward

    def set_cost_history(self, episode, cost):
        self.cost_history[episode] = cost

    def get_cost_history(self):
        return self.cost_history

    def update_histories(self, episode, step, action, reward, state):
        self.set_pos_history(episode, step, state[:2])
        self.set_vel_history(episode, step, state[2:])
        self.set_action_history(episode, step, action)
        self.set_reward_history(episode, step, reward)
        self.set_cost_history(episode, self.compute_distance(state))

    def get_histories(self):
        return {
            "position": self.position_history,
            "velocities": self.vel_history,
            "actions": self.actions_history,
            "reward": self.reward_history,
            "cost": self.cost_history,
        }

    def get_best_results(self):
        return self.best_results

    def compute_distance(self, state):
        """Compute the distance between the end effector and the target."""
        p = self.agent.h_q(state[0], state[1])[2:]
        return norm(p - self.target)

    def get_reward(self, state, episode):
        """
        Returns the reward for the current state.
        """
        # compute the distance between the end effector and the target
        missing_distance = self.compute_distance(state)
        residual_vel = norm(state[2:])
        # compute the reward
        reward = (missing_distance) + residual_vel #self.get_r_scale() *
        # get last reward
        nonzero_index = np.nonzero(self.reward_history[episode])[0]
        if len(nonzero_index) > 0:
            if reward >= abs(self.reward_history[episode, nonzero_index[-1]]):
                reward = reward * 100
            else:
                reward = 1
        # return the reward
        return -reward

    def is_done(self):
        """
        Checks if the current state is terminal.
        """
        cond = self.compute_distance(self.agent.get_state()) <= c.SIGMA
        if cond:
            self.agent.set_achieved_target(cond)
            return True
        self.agent.set_achieved_target(cond)
        return False

    def reset(self):
        if self.exploring_start:
            # randomize initial joint angles
            while True:
                q1 = np.random.uniform(c.WORKING_RANGE[0, 0], c.WORKING_RANGE[0, 1])
                q2 = np.random.uniform(c.WORKING_RANGE[1, 0], c.WORKING_RANGE[1, 1])
                if (self.agent.h_q(q1, q2)[2:] != self.target).all():
                    break
            self.agent.set_state(
                np.asarray(
                    [
                        q1,
                        q2,
                        0,
                        0,
                    ],
                    dtype=np.float64,
                )
            )
        else:
            self.agent.set_state(c.init_cond)
            self.agent.set_achieved_target(False)
        return self.agent.get_state()

    def get_active_tiles(self, state, action=[]):
        """
        The function `get_active_tiles` takes a state and an action as input and returns the indices of
        the active tiles based on the state variables and scaling factors.

        :param state: The state parameter is a list containing the values of the state variables. In
        this case, it contains four elements:
        :param action: The "action" parameter represents the action taken in the current state. It is
        used as an input to the "tiles" function to determine the active tiles
        :return: the indices of the active tiles.
        """
        # extract the state variables
        q1 = state[0]
        q2 = state[1]
        q1_d = state[2]
        q2_d = state[3]
        # define the scaling factor
        q1_sf = c.NUM_TILINGS * q1 / (c.WORKING_RANGE[0, 1] - c.WORKING_RANGE[0, 0])
        q2_sf = c.NUM_TILINGS * q2 / (c.WORKING_RANGE[1, 1] - c.WORKING_RANGE[1, 0])
        q1_d_sf = c.NUM_TILINGS * q1_d / (c.WORKING_VELOCITIES[0, 1] - c.WORKING_VELOCITIES[0, 0])
        q2_d_sf = c.NUM_TILINGS * q2_d / (c.WORKING_VELOCITIES[1, 1] - c.WORKING_VELOCITIES[1, 0])
        # get the indices of the active tiles
        active_tiles = tiles(self.iht, c.NUM_TILINGS, [q1_sf, q2_sf, q1_d_sf, q2_d_sf], action)
        return active_tiles

    def get_weights(self):
        return self.W

    def set_weights(self, weights):
        self.W = weights

    def get_active_weights(self, action_index: int, active_tiles: list):
        return self.W[action_index, active_tiles]

    def epsGreedy(self, state, episode, init=True):
        """The `epsGreedy` function implements an epsilon-greedy policy for selecting actions in a
        reinforcement learning algorithm.

        :param state: The "state" parameter represents the current state of the environment. It is used
        to determine the action to take based on the epsilon-greedy policy
        :param init: The `init` parameter is a boolean flag that indicates whether the `epsGreedy`
        function is being called for the first time or not. It is used to determine whether to explore
        (choose a random action) or exploit (choose the action with the highest Q-value) the current
        state, defaults to True (optional)
        :return: an action.
        """
        # get action with probability epsilon or if it is the first time
        #
        if np.random.rand() <= self.computeEps(episode) or init:
            index = np.random.randint(0, c.NUM_ACTIONS)
            return self.agent.get_selected_action(index)
        else:
            q_values = np.zeros((c.NUM_ACTIONS, 1))
            for i in range(c.NUM_ACTIONS):
                # get active tiles
                active_tiles = self.get_active_tiles(state, self.agent.get_selected_action(i))
                # compute the value of the current state-action pair
                q_values[i] = np.sum(self.W[i, active_tiles])
            # q_values = np.sum(self.get_weights())
            a_max = np.argmax(q_values)
        return self.agent.get_selected_action(a_max)

    def computeEps(self, episode):
        # compute the epsilon value based on the current episode number
        return c.EPS_START * np.exp(-c.EPS_DECAY * episode)

    def update(self, state, action, reward, next_state, next_action):
        """
        The function updates the weights of a SARSA agent based on the observed reward and the
        estimated value of the current and next state-action pairs.

        :param state: The current state of the environment. It represents the current situation or
        configuration of the system being modeled
        :param action: The "action" parameter represents the action taken in the current state
        :param reward: The reward is a scalar value that represents the immediate reward received after
        taking the action in the current state
        :param next_state: The parameter "next_state" represents the state that the agent transitions to
        after taking the current action. It is the state that the agent observes immediately after
        performing the action
        :param next_action: The parameter "next_action" represents the action taken in the next state.
        It is used to compute the value of the next state-action pair in order to update the weights
        """
        # get active tiles
        active_tiles = self.get_active_tiles(state, action)
        # compute the value of the current state-action pair
        a_i = np.where((self.agent.get_actions() == action).all(axis=1))[0][0]
        q_value = np.sum(self.get_active_weights(a_i, active_tiles))
        # get active tiles for the next state
        next_active_tiles = self.get_active_tiles(next_state, next_action)
        # compute the value of the next state-action pair
        next_a_i = np.where((self.agent.get_actions() == next_action).all(axis=1))[0][0]
        next_q_value = np.sum(self.get_active_weights(next_a_i, next_active_tiles))
        # compute the temporal difference error
        delta = reward + (c.GAMMA * next_q_value) - q_value
        if np.isnan(delta):
            print("Delta is NaN")
        # update the weights for the current state-action pair
        self.W[a_i, active_tiles] += c.ALPHA * delta
        if np.isnan(self.W).any():
            print("Weights are NaN")

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
        next_state = self.agent.step(state, action)
        # update the agent's state
        self.agent.set_state(next_state)
        # compute the reward
        reward = self.get_reward(state, episode)
        return next_state, reward

    def episode_loop(self, episode):
        """
        The function `episode_loop` is a loop that runs an episode of an agent interacting with an
        environment, updating the agent's weights and histories, and saving the weights and internal hash
        table on disk when the episode is done.

        :param agent: The "agent" parameter is an instance of a class that represents the agent in the
        reinforcement learning environment. It is responsible for interacting with the environment,
        selecting actions, and updating its state based on the rewards received
        :param episode: The parameter "episode" represents the current episode number in the training
        process. It is used to keep track of the progress and to save the weights and intermediate
        history tables at the end of each episode
        """
        # reset the agent
        self.reset()
        # get the initial state
        state = self.agent.get_state()
        # fill the first row of the histories
        self.update_histories(episode, 0, [0, 0], self.get_reward(state, episode), state)
        # get the initial action
        action = self.epsGreedy(state, episode)
        # loop until the episode is done
        for step in range(1, c.MAX_STEPS + 1):
            # get the next state and reward
            next_state, reward = self.get_state_reward_transition(state, action, episode)
            # get the next action
            next_action = self.epsGreedy(next_state, episode, False)
            # print(f"next action: {next_action}")
            # update the weights
            self.update(state, action, reward, next_state, next_action)
            # update the state and action
            state = next_state
            action = next_action
            # update agent's histories
            self.update_histories(episode, step, action, reward, state)
            # checks if the episode is done
            if self.is_done():
                print(f"Episode {episode} finished in {step} steps.")
                break
            elif step == c.MAX_STEPS:
                # if the episode is not done after the maximum number of steps, increase the penalty
                self.set_r_scale(self.get_r_scale() * c.R_SCALE)
        return step

    def SARSA_learning(self):
        """
        The function `SARSA_learning` implements the SARSA learning algorithm. It runs a loop that
        iterates over the episodes and calls the `episode_loop` function to run each episode.
        """
        assert self.agent is not None, "Agent not initialized"
        print("Learning started")
        for episode in range(c.NUM_EPISODES):
            print('\r', f"Episode: {episode} started...\t", end='', flush=True)
            steps = self.episode_loop(episode)
            if self.is_done():
                print(f"Episode {episode} finished in {steps} steps, saving results on disk...", end='')
                # save the weights and the iht on disk
                with open(f"data/weights/weights_{episode}.pkl", "wb") as f:
                    pickle.dump(self.W, f)
                with open(f"data/iht/iht_{episode}.pkl", "wb") as f:
                    pickle.dump(self.iht, f)
                with open(f"data/results/results_{episode}.pkl", "wb") as f:
                    pickle.dump(self.get_histories(), f)
                print(f"Done.\n")
            else:
                print(
                    f"not finished after {steps} steps, increasing penalty."
                    + f" The cumulative reward is {np.sum(self.reward_history[episode]):.4f}"
                )
        print('\n', "Learning finished")
        # save the results on disk


# Test the learning algorithm and plot the results
import matplotlib.pyplot as plt
import os
import re

if __name__ == "__main__":
    # initialize the agent
    arm = agent()
    # initialize the environment
    env = environ(arm)
    # set the agent's environment
    arm.set_env(env)
    learn = 2
    if learn == 0:
        env.SARSA_learning()
    elif learn == 1:
        data_dir = 'data'
        weights_dir = os.path.join(data_dir, 'weights')
        iht_dir = os.path.join(data_dir, 'iht')
        results_dir = os.path.join(data_dir, 'results')
        subdirs = [weights_dir, iht_dir, results_dir]
        weights = []
        ihts = []
        results = []
        files = [weights, ihts, results]
        episode_numbers = []
        for dir in subdirs:
            for file in os.listdir(dir):
                if file.endswith(".pkl"):
                    filename = os.path.join(dir, file)
                    with open(filename, "rb") as f:
                        files[subdirs.index(dir)].append(pickle.load(f))
                        if dir == weights_dir:
                            episode_numbers.append(int(re.findall(r'\d+', file)[0]))

        # assign the weights, the iht and the results of the final state
        w = weights[-1]
        iht = ihts[-1]
        res = results[-1]

        env.set_weights(w)
        env.set_iht(iht)

    # get the episode number

    # plot setup
    # fig, ax = plt.subplots(3, 2, figsize=(3, 7))
    fig, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'c'], ['d', 'e']], layout='constrained')
    ax['a'].set_title("visited q1-q2 pairs globally")
    ax['b'].axis("equal")
    ax['c'].set_title("Cost Function History")
    ax['d'].set_title("Control Law - U1")
    ax['e'].set_title("Control Law - U2")
    workspace = arm.generate_ws_curves()
    t_xx = c.TARGET[0] + c.SIGMA * np.cos(np.linspace(0, 2 * np.pi, 100))
    t_yy = c.TARGET[1] + c.SIGMA * np.sin(np.linspace(0, 2 * np.pi, 100))
    t = np.column_stack((t_xx, t_yy))
    # compute the initial arm configuration
    num_episodes = c.NUM_EPISODES if learn == 2 else episode_numbers[-1]
    for episode in range(num_episodes + 1):
        print(episode)
        if learn == 2:
            step = env.episode_loop(episode)
            res = env.get_histories()
        pos = res["position"][episode]
        cost = res["cost"][np.nonzero(res["cost"])[0]]
        x1, y1, x2, y2 = env.get_agent().h_q(pos[0, 0], pos[0, 1])
        # plot the visited q1-q2 pairs - update the plot at every episode
        ax['a'].plot(pos[:, 0], pos[:, 1], "o", markersize=1)
        # plot the trajectory - update the plot every 10 episodes
        if episode % 5 == 0 or episode == num_episodes + 1:
            # drop the elements that are zero (not visited)
            index = np.logical_and(pos[:, 0] != 0, pos[:, 1] != 0)
            index[0] = True
            pos = pos[index]
            # compute end effector trajectory for the whole story
            ef = env.get_agent().h_q(pos[:, 0], pos[:, 1])[2:]
            # clear the plot from the last shown trajectory
            ax['b'].cla()
            # plot the workspace
            ax['b'].plot(workspace[0], workspace[1], "m.", markersize=0.01)
            # plot the target
            ax['b'].plot(t[:, 0], t[:, 1], "-r*", markersize=1)
            # plot the arm's initial configuration
            ax['b'].plot([0, x1, x2], [0, y1, y2], "k--", linewidth=1.5)
            # plot the trajectory
            ax['b'].plot(ef[0, :], ef[1, :], "-b", markersize=0.5)
            # plot starting point
            ax['b'].plot(ef[0, 0], ef[1, 0], "go", markersize=5)
            # plot last point
            ax['b'].plot(ef[0, -1], ef[1, -1], "yo", markersize=5)
            # set title and legend
            ax['b'].set_title(f"End-Effector trajectory at episode {episode}")
            ax['b'].legend(["workspace", "target", "arm", "trajectory", "starting point", "last point"])
            # plot the cost function history
            ax['c'].cla()
            ax['c'].plot(range(len(cost)), cost, "-b", markersize=0.5)
            u1_set = res["actions"][episode][:, 0]
            u2_set = res["actions"][episode][:, 1]
            ax['d'].cla()
            ax['e'].cla()
            ax['d'].plot(range(len(u1_set)), u1_set, '*', label='u1')
            ax['d'].legend()
            ax['e'].plot(range(len(u2_set)), u2_set, '*', label='u2')
            ax['e'].legend()
            if episode < num_episodes + 1:
                plt.show(block=False)
                plt.pause(0.001)
            else:
                plt.show(block=False)
                plt.pause(3600)
                print('Done.')

    # if env.is_done():
    #    plt.pause()
    # print(
    #    f"not finished after {steps} steps, increasing penalty."
    #    + f" The cumulative reward is {np.sum(env.reward_history[episode])}"
    # )

    # get the histories
    # histories = env.get_histories()
    ## get the best results
    # results = env.get_best_results()
    ## plot the results
    # print(results)
