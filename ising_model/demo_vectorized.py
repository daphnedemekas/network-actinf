#%%

import os
from pathlib import Path

try:
    from simulation import Simulation, SimulationVectorized
except:
    from ising_model.simulation import Simulation, SimulationVectorized
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def run():

    N = 512
    k = 4
    p_ws = 0.0
    T = 1000

    # G = nx.watts_strogatz_graph(N,k,p_ws)

    num_cliques = 16
    nodes_per_clique = 32
    G = nx.ring_of_cliques(num_cliques, nodes_per_clique)
    N = G.number_of_nodes()

    G = nx.fast_gnp_random_graph(N, p=0.01)
    A = nx.to_numpy_array(G)

    # sampled_omegas = np.tril(np.random.rand(N, N)) # no diagonal i.e. no self-connections
    # sampled_omegas = np.tril(np.random.uniform(0.5,1.0,size=(N,N)))
    # omegas = sampled_omegas + np.tril(sampled_omegas,-1).T # make a symmetric matrix of omega parameters
    omegas = np.random.uniform(0.5, 0.7, size=(N, N))
    k_matrix = np.random.uniform(0.9, 1.1, size=(N, N))

    # sampled_omegas = 0.66666 * np.tril(np.ones((N, N))) # no diagonal i.e. no self-connections
    # omegas = sampled_omegas + np.tril(sampled_omegas, -1).T # make a symmetric matrix of omega parameters

    # ps_vec = np.random.rand(N)
    ps_vec = 0.5 * np.ones(N)

    sim = SimulationVectorized(G=G, k_matrix=k_matrix, p_s_vec=ps_vec)

    phi_hist, spin_hist = sim.run(T)
    # print(phi_hist.shape,phi_hist)
    # print(spin_hist.shape,spin_hist)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=100)

    until_T = T

    ax[0].imshow(phi_hist[:, :until_T], aspect="auto", interpolation="none")
    ax[0].set_title("Phi Hist")
    ax[1].imshow(spin_hist[:, :until_T], aspect="auto", interpolation="none")
    ax[1].set_title("Spin Hist")

    write_folder = Path(__file__).parent.parent.parent / "tmp_out"
    if not write_folder.is_dir():
        os.mkdir(write_folder)

    plt.savefig(os.path.join(write_folder, "tmp.png"), dpi=325, bbox_inches="tight")
    # plt.show()


def run_learning():

    N = 512
    k = 4
    p_ws = 0.0
    T = 1000

    # G = nx.watts_strogatz_graph(N,k,p_ws)

    num_cliques = 16
    nodes_per_clique = 32
    G = nx.ring_of_cliques(num_cliques, nodes_per_clique)
    N = G.number_of_nodes()

    G = nx.fast_gnp_random_graph(N, p=0.01)
    A = nx.to_numpy_array(G)

    # sampled_omegas = np.tril(np.random.rand(N, N)) # no diagonal i.e. no self-connections
    # sampled_omegas = np.tril(np.random.uniform(0.5,1.0,size=(N,N)))
    # omegas = sampled_omegas + np.tril(sampled_omegas,-1).T # make a symmetric matrix of omega parameters
    k_matrix = np.random.uniform(0.9, 1.1, size=(N, N))

    # sampled_omegas = 0.66666 * np.tril(np.ones((N, N))) # no diagonal i.e. no self-connections
    # omegas = sampled_omegas + np.tril(sampled_omegas, -1).T # make a symmetric matrix of omega parameters

    # ps_vec = np.random.rand(N)
    ps_vec = 0.5 * np.ones(N)

    sim = SimulationVectorized(G, k_matrix, p_s_vec=ps_vec, init_scale=1.0)

    phi_hist, spin_hist, k_matrix_hist = sim.run_learning(T)
    # print(phi_hist.shape,phi_hist)
    # print(spin_hist.shape,spin_hist)

    focal_agents = [0, 1]
    observed_agents = [3, 4]

    fig, axes = plt.subplots(
        len(focal_agents), len(observed_agents), figsize=(10, 4), dpi=100
    )

    until_T = T

    for i, agent_i in enumerate(focal_agents):
        for j, agent_j in enumerate(observed_agents):
            axes[i, j].plot(k_matrix_hist[agent_i, agent_j, :until_T])
            axes[i, j].plot(2 * spin_hist[agent_j, :until_T])
    write_folder = Path(__file__).parent.parent.parent / "tmp_out"
    if not write_folder.is_dir():
        os.mkdir(write_folder)

    plt.savefig(os.path.join(write_folder, "k_tmp.png"), dpi=325, bbox_inches="tight")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=100)

    ax[0].imshow(phi_hist[:, :until_T], aspect="auto", interpolation="none")
    ax[0].set_title("Phi Hist")
    ax[1].imshow(spin_hist[:, :until_T], aspect="auto", interpolation="none")
    ax[1].set_title("Spin Hist")

    write_folder = Path(__file__).parent.parent.parent / "tmp_out"

    plt.savefig(os.path.join(write_folder, "tmp.png"), dpi=325, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":

    run()
# run_learning()
