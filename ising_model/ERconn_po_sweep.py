from config import graph_generation_fns as gen_funcs
from simulation import Simulation

import os
import argparse
import itertools

import networkx as nx
import numpy as np

import pickle

graph_str = 'ER'
ps = 0.5

def run_sweep(param_combos, num_trials, T, run_name):

    for (param_idx, param_combo_i) in enumerate(param_combos):
        
        N, p, po = param_combo_i

        param_folder = os.path.join(run_name, f'{param_idx}')
        os.makedirs(param_folder)

        avg_vfe_per_trial = np.empty(num_trials)
        avg_complexity_per_trial = np.empty(num_trials)
        avg_neg_accur_per_trial = np.empty(num_trials)
        avg_polarization_per_trial = np.empty(num_trials)
        avg_m_per_trial = np.empty(num_trials)
        adj_mat_per_trial = np.zeros((N,N, num_trials))

        for trial_i in range(num_trials):
            G = gen_funcs[graph_str](N, p) 

            sim = Simulation(G)

            adj_mat_per_trial[:,:,trial_i] = sim.A

            phi_hist, spin_hist = sim.run(T, po, ps)

            vfe, complexity, neg_accur = sim.compute_VFE(phi_hist, spin_hist, decomposition="complexity_accuracy")

            avg_vfe_per_trial[trial_i] = vfe[:,100:].mean() # exclude transient
            avg_complexity_per_trial[trial_i] = complexity[:,100:].mean()  # exclude transient
            avg_neg_accur_per_trial[trial_i] = neg_accur[:,100:].mean()  # exclude transient
            avg_polarization_per_trial[trial_i] = phi_hist[:,100:].mean() # exclude transient
            avg_m_per_trial[trial_i] = sim.compute_m(phi_hist[:,100:])

        results_dict = {'avg_vfe_per_trial': avg_vfe_per_trial,
                        'avg_complexity_per_trial': avg_complexity_per_trial,
                        'avg_neg_accur_per_trial': avg_neg_accur_per_trial,
                        'avg_polarization_per_trial': avg_polarization_per_trial,
                        'avg_m_per_trial': avg_m_per_trial,
                        'adj_mat_per_trial': adj_mat_per_trial,
                        'N': N, 
                        'p': p, 
                        'po':po}

        with open(os.path.join(param_folder, "results.pkl"), 'wb') as output:
            # Pickle dictionary using protocol 0.
            pickle.dump(results_dict, output)

        # for future reference, when it comes to visualizing/analyzing the data
        # # load data from pkl file
        # with open(os.path.join(param_folder, "results.pkl"), "rb") as fp:
        #     results_dict = pickle.load(fp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", "-name", type = str,
                        help = "Name of job or run",
                        dest = "run_name", default = "default_run_name")
    parser.add_argument("--num_trials", "-nt", type = int,
                        help = "Number of trials to run per parameter configuration",
                        dest = "num_trials", default = 200)
    parser.add_argument("--time_limit", "-T", type = int,
                        help = "Number of timesteps to run per simulation",
                        dest = "T", default = 1000)
    parser.add_argument("--network_size", "-N", type=int, 
                        help = "Size of network",
                        dest = "N")

    run_args = vars(parser.parse_args())

    run_name = run_args["run_name"]
    if not os.path.exists(run_name):
        os.makedirs(run_name)
    
    num_trials, N, T = run_args["num_trials"], run_args["N"], run_args["T"]

    starting_p = 1.5 * np.log(N)/N # log(N)/N is the value of p where network is connected about 36% of the time
    ER_p_levels = np.linspace( starting_p, 1.0, 20)
    po_levels = np.linspace( 0.5 + 1e-16, 1.0 - 1e-16, 50)

    param_combos = itertools.product([N], ER_p_levels, po_levels)

    run_sweep(param_combos, num_trials, T, run_name)


  





