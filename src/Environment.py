import constants as c
import numpy as np
from tiles3 import IHT, tiles
from agent import agent
from numpy.linalg import norm
import pickle


class environ(object):
    def __init__(self, agent=None):
        self.iht = IHT(c.NUM_FEATURES)
        # self.W = np.zeros((c.NUM_ACTIONS, c.NUM_FEATURES))
        self.W = np.random.rand(c.NUM_ACTIONS, c.NUM_FEATURES)
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
        self.agent = agent
        self.best_results = {'positions': [], 'actions': [], 'rewards': []}

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

    def epsGreedy(self, state, init=True):
        """
        The `epsGreedy` function implements an epsilon-greedy policy for selecting actions in a
        reinforcement learning algorithm.

        :param state: The "state" parameter represents the current state of the environment. It is used
        to determine the action to take based on the epsilon-greedy policy
        :param init: The `init` parameter is a boolean flag that indicates whether the `epsGreedy`
        function is being called for the first time or not. It is used to determine whether to explore
        (choose a random action) or exploit (choose the action with the highest Q-value) the current
        state, defaults to True (optional)
        :return: an action.
        """
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
        a_i = np.where((self.actions == action).all(axis=1))[0][0]
        q_value = np.sum(self.W[a_i, active_tiles])
        # get active tiles for the next state
        next_active_tiles = self.get_active_tiles(next_state, next_action)
        # compute the value of the next state-action pair
        a_i_next = np.where((self.actions == next_action).all(axis=1))[0][0]
        next_q_value = np.sum(self.W[a_i_next, next_active_tiles])
        # update the weights
        delta = reward + c.GAMMA * next_q_value - q_value
        self.W[a_i, active_tiles] += c.ALPHA * delta

    def episode_loop(self, agent, episode):
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
        agent.reset()
        # get the initial state
        state = agent.get_state()
        # get the initial action
        action = self.epsGreedy(state)
        # loop until the episode is done
        for step in range(c.MAX_STEPS):
            # get the next state and reward
            next_state, reward = agent.get_state_reward_transition(state, action, episode)
            # get the next action
            next_action = self.epsGreedy(next_state, False)
            # update the weights
            self.step_update(state, action, reward, next_state, next_action)
            # update agent's histories
            agent.update_histories(episode, step, action, reward)
            # checks if the episode is done
            if agent.is_done():
                print(f"Episode {episode} finished in {step} steps")
                # save the weights and the iht on disk
                with open("../data/weights.pkl", "wb") as f:
                    pickle.dump(self.W, f)
                with open("../data/iht.pkl", "wb") as f:
                    pickle.dump(self.iht, f)
                print(f"Episode {episode} weights and iht saved")
                break
            elif step == c.MAX_STEPS - 1:
                print(f"Episode {episode} not finished, increasing penalty")
                # increase the penalty of 10% - what is zero in the beginning will remain zero
                self.W *= 1.5
            # update the state and action
            state = next_state
            action = next_action

    def SARSA_learning(self):
        """
        The function `SARSA_learning` implements the SARSA learning algorithm. It runs a loop that
        iterates over the episodes and calls the `episode_loop` function to run each episode.
        """
        assert self.agent is not None, "Agent not initialized"
        print("Learging started")
        for episode in range(c.NUM_EPISODES):
            print('\r', f"Episode: {episode}", end='', flush=True)
            self.episode_loop(self.agent, episode)
        print('\n', "Learning finished")
        # fill the best results dictionary
        self.best_results['positions'] = self.agent.position_history[episode]
        self.best_results['actions'] = self.agent.action_history[episode]
        self.best_results['rewards'] = self.agent.reward_history[episode]

    def get_best_results(self):
        return self.best_results
