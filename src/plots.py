from cProfile import label
import matplotlib.pyplot as plt
from agent import Agent
from environment import Environment
import pickle
import constants as c
import numpy as np
from numpy.linalg import norm

# load the results from the pickle file
results = pickle.load(open("data/results/results_999.pkl", "rb"))
# load the weights from the pickle file
weights = pickle.load(open("data/weights/weights_999.pkl", "rb"))

# create the agent
arm = Agent()
# create the environment
env = Environment(arm)
env.set_weights(weights)
# get the results
positions = results["position"]
velocities = results["velocities"]
actions = results["actions"]
reward = results["reward"]
costs = results["cost"]
# get the number of episodes and steps
num_eps = c.NUM_EPISODES
num_steps = c.MAX_STEPS
# compute the cumulative reward
cum_rew = np.sum(reward, axis=1)
### plot the results ###
fig, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'd']], layout='constrained')
ax['a'].set_title("visited q1-q2 pairs globally")
ax['b'].axis("equal")
ax['c'].set_title("Cost Function History (remain distance at the end of each episode))")
ax['d'].set_title("Comulative Reward")
# generate workspace and target
workspace = arm.generate_ws_curves()
t_xx = c.TARGET[0] + c.SIGMA * np.cos(np.linspace(0, 2 * np.pi, 100))
t_yy = c.TARGET[1] + c.SIGMA * np.sin(np.linspace(0, 2 * np.pi, 100))
t = np.column_stack((t_xx, t_yy))


for eps in range(num_eps):
    # get datas for the current episode
    pos = positions[eps]
    # filter the data when they are still zero (episode finished before max steps)
    index = np.logical_and(pos[:, 0] != 0, pos[:, 1] != 0)
    index[0] = True
    pos = pos[index]
    vel = velocities[eps][: len(pos)]
    a = actions[eps][: len(pos)]
    rew = reward[eps][: len(pos)]
    # plot the visited q1-q2 pairs globally
    ax['a'].plot(pos[:, 0], pos[:, 1], "-o", markersize=1)
    if (eps % 200 == 0 or eps == c.NUM_EPISODES-1) and eps != 0:

        # compute initial arm configuration for the current episode
        x1, y1, x2, y2 = env.get_agent().h_q(pos[0, 0], pos[0, 1])
        # compute end effector trajectory for the whole story
        ef = env.get_agent().h_q(pos[:, 0], pos[:, 1])[2:]
        # clear the axes
        ax['b'].cla()
        # plot the workspace
        ax['b'].plot(workspace[0], workspace[1], "m.", markersize=0.01, label="workspace")
        # plot the target
        ax['b'].plot(t[:, 0], t[:, 1], "-r*", markersize=1, label="target")
        # plot the arm's initial configuration
        ax['b'].plot([0, x1, x2], [0, y1, y2], "k--", linewidth=1.5, label="arm")
        # plot the trajectory
        ax['b'].plot(ef[0, :], ef[1, :], "-b", markersize=0.5, label="trajectory")
        # plot starting point
        ax['b'].plot(ef[0, 0], ef[1, 0], "go", markersize=5, label="starting point")
        # plot last point
        ax['b'].plot(ef[0, -1], ef[1, -1], "yo", markersize=5, label="last point")
        # set title and legend
        ax['b'].set_title(f"End-Effector trajectory at episode {eps}")
        ax['b'].legend()
        # plot the cost function history
        ax["c"].cla()
        ax['c'].plot(range(eps), costs[:eps], "-b", markersize=0.5, label="dist. to target")
        ax['c'].legend()
        # plot the cumulative reward history
        ax["d"].cla()
        ax['d'].plot(range(eps), cum_rew[:eps] / c.NUM_EPISODES, "-b", markersize=0.5, label="cumulative reward")
        ax['d'].legend()
        # every 20 episodes update the other plots
        if eps % 200 == 0:
            plt.show(block=False)
            plt.pause(0.001)
        if eps == c.NUM_EPISODES - 1:
            plt.show(block=False)
            plt.pause(0.001)
            print('done')

# try to reach the target with the learned policy
# fig, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'd']], layout='constrained')
# ax['a'].set_title("Arm Trajectory")
# ax['a'].axis("equal")
# ax['b'].set_title("Cumulative Reward")
# ax['c'].set_title("Control Law - U1")
# ax['d'].set_title("Control Law - U2")
## start the simulation
# w = env.get_weights()
# is_terminal = False
# pred_reward = 0
# step = 0
# reward_hist = np.empty((0, 1))
# state_hist = np.empty((0, 4))
# action_hist = np.empty((0, 2))
# target = np.concatenate((c.TARGET, np.zeros(2)))
# while not is_terminal:
#    # get the current state
#    state = arm.get_state()
#    # Compute Q-values for all actions
#    X = [env.get_features(state, env.agent.get_selected_action(i)) for i in range(c.NUM_ACTIONS)]
#    Q = np.array([np.sum(w[x == 1]) for x in X])
#    # select the action with the highest Q-value
#    action = env.agent.get_selected_action(np.argmax(Q))
#    # take the action
#    next_state = env.agent.step(state, action)
#    env.agent.set_state(next_state)
#    # compute the reward
#    reward = -norm(state - target)
#    if pred_reward != 0:
#        # penalize the reward as in learning phase if the prediction is wrong
#        # for consistency with the learning phase
#        reward = reward * 1e3 if reward <= pred_reward else reward
#    pred_reward = reward
#    # update histories
#    state_hist = np.vstack((state_hist, state))
#    reward_hist = np.vstack((reward_hist, reward))
#    action_hist = np.vstack((action_hist, action))
#    # update the environment
#    step += 1
#    state = next_state
#    # check if the episode is finished
#    is_terminal = env.is_done()
#
# for i in range(0, len(state_hist)):
#    # plot the arm's trajectory
#    x1, y1, x2, y2 = env.get_agent().h_q(state_hist[i, 0], state_hist[i, 1])
#    ax["a"].cla()
#    # plot the workspace
#    ax['a'].plot(workspace[0], workspace[1], "m.", markersize=0.01, label="workspace")
#    # plot the target
#    ax['a'].plot(t[:, 0], t[:, 1], "-r*", markersize=1, label="target")
#    # plot the arm's initial configuration
#    ax['a'].plot([0, x1, x2], [0, y1, y2], "k--", linewidth=1.5, label="arm")
#    # plot last point
#    ax['a'].plot(x2[-1], y2[-1], "yo", markersize=5, label="last point")
#
