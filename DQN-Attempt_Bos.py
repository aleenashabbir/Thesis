# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import networkx as nx
import osmnx as ox
import pickle
#import wandb
from collections import namedtuple, deque
import os
import numpy as np
#os.environ["WANDB_API_KEY"] = "c3dfd411807005336933dfdbb175bcb97e13b624"
#os.environ["WANDB_MODE"] = "online"
from tqdm import tqdm
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[0]
    return av_act

def sample_next_action(available_actions_range, previous_state, state):
    if available_actions_range.size == 0:
        #print("reached")
        available_actions_range = np.setdiff1d(available_actions(previous_state), state)
        print(available_actions_range)
        next_action = int(np.random.choice(available_actions_range,1))
    else:
        next_action = int(np.random.choice(available_actions_range,1))
    return next_action


# def update(current_state, action, gamma):
#     # print(action)
#     gamma = 0.965
#     max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
#     # print(max_index)
#
#     if max_index.shape[0] > 1:
#         max_index = int(np.random.choice(max_index, size=1))
#     else:
#         max_index = int(max_index)
#
#     max_value = Q[action, max_index]
#
#     Q[current_state, action] = R[current_state, action] + gamma * max_value
#     # print('max_value', R[current_state, action] + gamma * max_value)
#
#     if (np.max(Q) > 0):
#         return (np.sum(Q / np.max(Q) * 100))
#     else:
#         return (0)


def itergdfbearing(gdf, val_checker, state, previous_state):
    bearing_val = []
    if len(val_checker) > 0:
        for j in val_checker:
            # print(gdf.iloc[j]['bearing'])
            # print((np.abs(180-gdf.iloc[j]['bearing'])))
            bearing_val.append((np.abs(180 - gdf.iloc[j]['bearing'])))
        if len(bearing_val) != 0:
            smallest_angle = min(bearing_val)
            placer = bearing_val.index(smallest_angle)
            index = val_checker[placer]
            next_node_id = gdf.iloc[index]['v']
            following_state = nodes_list.index(next_node_id)
        else:
            no_edge = []
            no_edge.append(state)
            available_act = np.setdiff1d(available_actions(previous_state), state)
            if available_act.size == 0:
                no_edge.append(previous_state)
                new = np.random.choice(np.setdiff1d(range(0, int(Q.shape[0])), no_edge))
                available_act = available_actions(new)
            following_state = sample_next_action(available_act, previous_state, state)

    else:
        available_act = np.setdiff1d(available_actions(previous_state), state)
        following_state = sample_next_action(available_act, previous_state, state)

    return following_state

def itergdfstate(gdf, state):
    val_checker = []
    node_id = nodes_list[state]
        # print(node_id)
    for i in range(0, len(gdf)):
            # node_id = nodes_list[state]
            # print(type(gdf.iloc[i]['u']))
        if gdf.iloc[i]['u'] == node_id:
            val_checker.append(i)
    return val_checker

def carolspicker(state, previous_state):
    indices = itergdfstate(gdf_edges_stack, state)
    next_action = itergdfbearing(gdf_edges_stack, indices, state, previous_state)
    return next_action

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

##################################STARTING ALGORITHM TRAINING

place_name = "Boston, United States"
graph = ox.graph_from_place(place_name)

nodes_list = list(graph.nodes())
adjmatrix = nx.to_numpy_array(graph)

nodec = ox.nearest_nodes(graph,-71.078096, 42.331018)
carol_pos = nodes_list.index(nodec)

nodeps = ox.nearest_nodes(graph, -71.089165, 42.351355)
paul_start = nodes_list.index(nodeps)

nodepe = ox.nearest_nodes(graph, -71.139223, 42.233716)
paul_end = nodes_list.index(nodepe)

graph = ox.add_edge_bearings(graph)
gdf_edges = ox.graph_to_gdfs(graph, nodes=False)
gdf_edges_stack = gdf_edges.reset_index()


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

MATRIX_SIZE = len(nodes_list)

# create matrix x*y
MATRIX_SIZE = len(nodes_list)
R = np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE))
R *= -1

adjmatrix = nx.to_numpy_array(graph)
adjmatrix

for i in range(0, len(adjmatrix)):
    for j in range(0, len(adjmatrix)):
        if adjmatrix[i][j] > 0.0:
            R[i][j] = 0
R[paul_end, paul_end] = 1000
R[carol_pos, carol_pos] = -1000000000000000000000000000000000000000000000000

# n_actions = available_actions(state)
# n_observations = len(R)
# policy_net = DQN(n_observations, n_actions).to(device)
# target_net = DQN(n_observations, n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())
#
# optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# memory = ReplayMemory(10000)


steps_done = 0
episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

range_pick = len(nodes_list)
listrange_pick = (list(i) for i in range(0, len(nodes_list)))
print(listrange_pick)
open_door = []
for i_episode in tqdm(range(1000)):
    # Initialize the environment and get it's state
    # if gym.__version__[:4] == '0.26':
    #     state, _ = env.reset()
    # elif gym.__version__[:4] == '0.25':
    #     state, _ = env.reset(return_info=True)
    #state = np.random.choice(np.setdiff1d(range(0, listrange_pick, open_door)))
    state = np.random.choice(np.setdiff1d(range(0, range_pick), open_door))
    n_actions = available_actions(state)
    while n_actions.size == 0:
        open_door.append(state)
        #np.random.choice(np.setdiff1d(range(0, int(Q.shape[0])), open_door))
        state = np.random.choice(np.setdiff1d(range(0, range_pick), open_door))
    n_observations = len(nodes_list)
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = available_actions(state)
        #observation, reward, terminated, truncated, _ = update(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
#wandb.init(proj="Boston_Q")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
