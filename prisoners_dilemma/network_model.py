#%%
import os

os.chdir("/Users/daphnedemekas/Desktop/Research/network-actinf/")
import networkx as nx
from network.networks import *
from prisoners_dilemma.utils import get_agent_params, pd_one_round
from pymdp.agent import Agent
import random

# %%
ER_network = nx.fast_gnp_random_graph(n=20, p=0.8)
pos = nx.spring_layout(ER_network)
ER_network.add_nodes_from(pos.keys())
for n, p in pos.items():
    ER_network.nodes[n]["pos"] = p
# %%

factors_to_learn = "all"
lr_pB = 0.1
import numpy as np


def setup_network_deterministic_fixed_lr(G):

    agents_dict = {}
    A, B, C, D, pB_1, pB_2 = get_agent_params()

    for i in G.nodes():
        agent_i = Agent(
            A=A, B=B, C=C, D=D, pB=pB_1, lr_pB=lr_pB, factors_to_learn=factors_to_learn
        )
        agent_i.observation = None
        agent_i.qs = D
        agents_dict[i] = agent_i

    nx.set_node_attributes(G, agents_dict, "agent")

    return G


G = setup_network_deterministic_fixed_lr(ER_network)


def run_simulation(G, T: int):
    """Runs a network simulation where agents play
    iterative prisoners dilemma with a random neighbour
    at each trial"""
    rounds = dict.fromkeys(range(T))
    for t in range(T):
        rounds[t] = dict.fromkeys(G.nodes())

        for i in G.nodes():
            if (
                rounds[t][i] is not None
            ):  # this agent has played as an opponent already in this round
                continue
            i_neighbours = list(nx.neighbors(G, i))
            o = random.choice(i_neighbours)

            agent_node_attrs = G.nodes()[i]
            opponent_node_attrs = G.nodes()[o]

            agent = agent_node_attrs["agent"]
            opponent = opponent_node_attrs["agent"]

            if agent.observation == None:
                observation_1 = [0]
            else:
                observation_1 = agent.observation

            if opponent.observation == None:
                observation_2 = [0]
            else:
                observation_2 = opponent.observation

            action_1, action_2, B1, B2, q_pi_1, q_pi_2, qs_1, qs_2 = pd_one_round(
                agent, opponent, observation_1, observation_2, t
            )

            rounds[t][i] = {
                "opponent": o,
                "action": action_1,
                "B": B1,
                "q_pi": q_pi_1,
                "q_s": qs_1,
            }
            rounds[t][o] = {
                "opponent": i,
                "action": action_2,
                "B": B2,
                "q_pi": q_pi_2,
                "q_s": qs_2,
            }

    return rounds
T = 20

rounds = run_simulation(G, T)

# %%
def get_actions(rounds, T, N):
    actions = np.zeros((T, N))
    for t in range(T):
        for i, round in enumerate(rounds[t].values()):
            actions[t, i] = round["action"]
    return actions


actions = get_actions(rounds, T, len(G.nodes()))
# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import imageio


def draw_actions(actions):
    ims = []
    fig = plt.figure()
    for trial in actions:
        colormap = []
        for action in trial:
            if int(action) == 0:
                colormap.append("green")
            else:
                colormap.append("blue")
        im = nx.draw(G, pos, node_color=colormap)
        plt.savefig("img.png")

        im = imageio.imread("img.png")
        ims.append(im)
    imageio.mimsave("networkgif.gif", ims)


# %%
