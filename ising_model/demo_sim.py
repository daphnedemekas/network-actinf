#%%

import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

#%%

N = 500 # number of nodes

p = 0.01 # connection probability of ER graph

# avg_connected = 0.
# n_iter = 1000
# for _ in range(n_iter):
#     G = nx.fast_gnp_random_graph(N, 0.01) # initialize ER graph
#     avg_connected += float(nx.is_connected(G))/n_iter

# k = 4 # number of links in WS graph
# p = 0. # re-wiring probability in WS graph
# G = nx.watts_strogatz_graph(N, 4, 0.)

# create a sqrt(N) x sqrt(N) size lattice (with/without periodic boundaries)
# d1 = int(np.sqrt(N))
# d2 = int(N / d1)
# N = d1 * d2 # re-adjust N to make it a square number
# G = nx.grid_2d_graph(d1, d2, periodic=True)

# get the adjacency matrix
A = nx.to_numpy_array(G)

# initialization option 1: sample spins to begin with, and then initialize posteriors to those spins
# initial_spins = 2*np.random.randint(2, size = (N,)) - 1 # in terms of (-1, 1)
# initial_spins = np.random.randint(2, size = (N,)) # in terms of (0, 1)
# posteriors = np.absolute((initial_spins - 1.)) # posterior is belief about being in down state, so we have to subtract 1.0 first

# initialization option 2: sample random posteriors in [0, 1] and then get initial spin-states/posteriors by threshold the posteriors (arg-maxing action-selection, essentially)
initial_posteriors = np.random.rand(N) # posterior is belief about belign in down state
initial_spins = (initial_posteriors > 0.5).astype(float) # if spin == 1, you're in a DOWN spin, otherwise, you're in an UP sstate

# %% Simulation

def run_simulation(T, initial_spins, initial_posteriors, A, po, ps):

    # calculate these in advance for quick use in VFE calculations
    logpo = np.log(po)
    logps = np.log(ps)
    logpo_C = np.log(1-po)
    logps_C = np.log(1-ps)

    # spin states -- 1.0 == DOWN, 0.0 == UP
    spin_state = initial_spins.copy()

    # posteriors
    phi = initial_posteriors.copy()

    # history of posteriors and spins
    spin_hist = np.zeros ((N, T) )
    phi_hist = np.zeros( (N, T) )

    # history of components of VFE (either negH - energy or kld - accur, depending on what you want to look at)
    # negH_hist = np.zeros( (N, T) )
    # energy_hist = np.zeros ((N, T) )
    kld_hist = np.zeros( (N, T) )
    accur_hist = np.zeros( (N, T) )

    for t in range(T):

        sum_down_spins = A @ spin_state # this sums up the neighbours that are DOWN, per agent

        up_spins = np.absolute(spin_state - 1) # converts from 1.0 meaning DOWN to 1.0 meaning UP
        sum_up_spins = A @ up_spins # this sums up the neighbours that are UP, per agent

        spin_diffs = sum_down_spins - sum_up_spins # difference in DOWNs vs UPs, per agent

        # equivalent expressions
        # x = ((ps - 1.) * (((1./po) - 1.)**spin_diffs)) / ps  
        x = (1. - (1./ps)) * ((1. - po)/po)**(spin_diffs)  
        phi = 1. / (1. - x)

        # sample new spin-state -- probability that I'm DOWN
        spin_state = (phi > np.random.rand(N)).astype(float)

        # store histories of spin states and posteriors
        spin_hist[:,t] = spin_state.copy()
        phi_hist[:,t] = phi.copy()

        # compute VFE for each agent (@NOTE: this could be computed outside this function, after the fact - probably should be done in order to speed things up)

        # get the other Bernoulli param (complement of phi)
        phi_C = 1. - phi

        # Decomposition 1: negative entropy - expected variational energy (uncomment below if you want to do this)
        # negH = phi * np.log(phi + 1e-16) + phi_C * np.log(phi_C + 1e-16)
        # neg_expected_energy = phi * (sum_down_spins * logpo + sum_up_spins*logpo_C + logps) + phi_C*(sum_up_spins*logpo + sum_down_spins*logpo_C + logps_C)
        
        # negH_hist[:,t] = negH
        # energy_hist[:,t] = -neg_expected_energy

        # Decomposition 2: complexity - accuracy (uncomment below if you want to do this)
        kld = phi * (np.log(phi + 1e-16) - logps) + phi_C * (np.log(phi_C + 1e-16) - logps_C)
        accur = phi * (sum_down_spins*logpo + sum_up_spins*logpo_C) + phi_C*(sum_up_spins*logpo + sum_down_spins*logpo_C)

        kld_hist[:,t] = kld
        accur_hist[:,t] = accur

    # return phi_hist, spin_hist, negH_hist, energy_hist
    return phi_hist, spin_hist, kld_hist, accur_hist

# %%

# initialization option 2: sample random posteriors in [0, 1] and then get initial spin-states/posteriors by threshold the posteriors (arg-maxing action-selection, essentially)
initial_posteriors = np.random.rand(N) # posterior is belief about belign in down state
initial_spins = (initial_posteriors > 0.5).astype(float) # if spin == 1, you're in a DOWN spin, otherwise, you're in an UP sstate

# G = nx.fast_gnp_random_graph(N, 0.01)
# A = nx.to_numpy_array(G)
phi_hist, spin_hist, _,_ = run_simulation(1000, initial_spins, initial_posteriors, A, 0.99, 0.5)
plt.imshow(spin_hist, aspect = 'auto', interpolation = 'none', cmap = 'gray')

# %% Plot heatmaps of "spiking" activity in terms of DOWN/UP states

N, T = 1000, 1000
G = nx.fast_gnp_random_graph(N, 0.01)
A = nx.to_numpy_array(G)
initial_posteriors = np.random.rand(N)
initial_spins = (initial_posteriors > 0.5).astype(float)
phi_hist, spin_hist, _,_ = run_simulation(T, initial_spins, initial_posteriors, A, 0.50001, 0.5)

fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (10, 8), sharex = True, sharey = False)
axes[0].imshow(spin_hist[::10,:], aspect = 'auto', interpolation = 'none', cmap = 'gray')
axes[0].set_xlim(0, T)
axes[0].tick_params(axis='both', which='major', labelsize=16)
axes[0].set_title('$p_{\mathcal{O}} = 0.5$', fontsize = 18)

G = nx.fast_gnp_random_graph(N, 0.01)
A = nx.to_numpy_array(G)
initial_posteriors = np.random.rand(N)
initial_spins = (initial_posteriors > 0.5).astype(float)
phi_hist, spin_hist, _,_ = run_simulation(T, initial_spins, initial_posteriors, A, 0.601, 0.5)

axes[1].imshow(spin_hist[::10,:], aspect = 'auto', interpolation = 'none', cmap = 'gray')
axes[1].set_xlim(0, T)
axes[1].tick_params(axis='both', which='major', labelsize=16)
axes[1].set_title('$p_{\mathcal{O}} = 0.61$', fontsize = 18)
axes[1].set_ylabel('lattice site $k \in \Lambda$', fontsize = 18)

G = nx.fast_gnp_random_graph(N, 0.01)
A = nx.to_numpy_array(G)
initial_posteriors = np.random.rand(N)
initial_spins = (initial_posteriors > 0.5).astype(float)
phi_hist, spin_hist, _,_ = run_simulation(T, initial_spins, initial_posteriors, A, 0.75, 0.5)

axes[2].imshow(spin_hist[::10,:], aspect = 'auto', interpolation = 'none', cmap = 'gray')
axes[2].set_xlim(0, T)
axes[2].set_xlabel('$t$', fontsize = 18)
axes[2].tick_params(axis='both', which='major', labelsize=16)
axes[2].set_title('$p_{\mathcal{O}} = 0.75$', fontsize = 18)

plt.savefig('different_regimes_individualhistories.png', dpi = 325)

#%% Plot three examples of dynamics -- random, near-critical, and polarized

T = 1000
initial_posteriors = np.random.rand(N)
initial_spins = (initial_posteriors > 0.5).astype(float)
phi_hist, spin_hist, _,_ = run_simulation(T, initial_spins, initial_posteriors, A, 0.50001, 0.5)

A_t = phi_hist.mean(axis = 0)

fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (10, 8), sharex = True, sharey = False)
axes[0].plot(np.arange(T), A_t)
axes[0].set_xlim(0, T)
axes[0].tick_params(axis='both', which='major', labelsize=16)
axes[0].set_title('$p_{\mathcal{O}} = 0.5$', fontsize = 18)

initial_posteriors = np.random.rand(N)
initial_spins = (initial_posteriors > 0.5).astype(float)
phi_hist, spin_hist, _,_ = run_simulation(T, initial_spins, initial_posteriors, A, 0.601, 0.5)

A_t = phi_hist.mean(axis = 0)

axes[1].plot(np.arange(T), A_t)
axes[1].set_xlim(0, T)
axes[1].tick_params(axis='both', which='major', labelsize=16)
axes[1].set_title('$p_{\mathcal{O}} = 0.61$', fontsize = 18)

initial_posteriors = np.random.rand(N)
initial_spins = (initial_posteriors > 0.5).astype(float)
phi_hist, spin_hist, _,_ = run_simulation(T, initial_spins, initial_posteriors, A, 0.75, 0.5)

A_t = phi_hist.mean(axis = 0)

axes[2].plot(np.arange(T), A_t)
axes[2].set_xlim(0, T)
axes[2].set_xlabel('$t$', fontsize = 18)
axes[2].set_ylabel('$A_t$', fontsize = 18)
axes[2].tick_params(axis='both', which='major', labelsize=16)
axes[2].set_title('$p_{\mathcal{O}} = 0.75$', fontsize = 18)

plt.savefig('different_regimes.png', dpi = 325)


#%% Run a sweep over values of po and measure average variational free energy
ps = 0.5

N, T = 500, 1000

po_vec = np.linspace(0.5, 0.999, 50)

n_trials = 10

vfe_all_trials = np.zeros( (n_trials, len(po_vec) ) )
kld_all_trials = np.zeros( (n_trials, len(po_vec) ) )
accur_all_trials = np.zeros( (n_trials, len(po_vec) ) )

for (ii, po) in enumerate(po_vec):

    for trial_i in range(n_trials):
        G = nx.fast_gnp_random_graph(N, 0.01)
        A = nx.to_numpy_array(G)

        initial_posteriors = np.random.rand(N)
        initial_spins = (initial_posteriors > 0.5).astype(float)
        phi_hist, spin_hist, kld_hist, accur_hist = run_simulation(T, initial_spins, initial_posteriors, A, po, ps)

        vfe_hist = kld_hist - accur_hist
    
        vfe_all_trials[trial_i,ii] = vfe_hist.mean()
        kld_all_trials[trial_i,ii] = kld_hist.mean()
        accur_all_trials[trial_i,ii] = accur_hist.mean()

#%% plot average value of VFE as a function of po, with confidence intervals

# fig, axes = plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize = (10, 6))
fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (10, 6))

std_val = 1.96 * vfe_all_trials.std(axis=0)
mean_val = vfe_all_trials.mean(axis=0)
axes[0].fill_between(po_vec, mean_val + std_val, mean_val - std_val, alpha = 0.5)
axes[0].plot(po_vec, mean_val, label = "$\mathcal{F}$")
axes[0].legend(loc = "lower right", fontsize=16)

axes[0].set_xticks(ticks = np.linspace(po_vec[0], 1.0, 6), fontsize = 16)
axes[0].set_yticks(ticks = np.linspace(-2.5, 5.0, 4), fontsize = 16)
axes[0].set_xlim((po_vec[0], po_vec[-2]))

std_val = 1.96 * kld_all_trials.std(axis=0)
mean_val = kld_all_trials.mean(axis=0)
axes[1].fill_between(po_vec, mean_val + std_val, mean_val - std_val, alpha = 0.5)
axes[1].plot(po_vec, mean_val, label = "Complexity")
axes[1].legend(loc = "lower right", fontsize=16)

axes[1].set_xticks(ticks = np.linspace(po_vec[0], 1.0, 6), fontsize = 16)
axes[1].set_yticks(ticks = np.linspace(-3., 3., 4), fontsize = 16)
axes[1].set_xlim((po_vec[0], po_vec[-2]))

std_val = 1.96 * accur_all_trials.std(axis=0)
mean_val = -accur_all_trials.mean(axis=0)
axes[1].fill_between(po_vec, mean_val+std_val, mean_val - std_val, alpha = 0.5)
axes[1].plot(po_vec, mean_val, label = "(-ve) Accuracy")
axes[1].legend(loc = "lower right", fontsize=16)

axes[1].set_xlabel('Likelihood parameter: $p_{\mathcal{O}}$', fontsize = 18)
axes[1].set_ylabel('Average nats', fontsize = 16)

axes[0].set_ylabel('$\hat{\mathcal{F}}$ (variational free energy)', fontsize = 16)

# std_val = 1.96 * accur_all_trials.std(axis=0)
# mean_val = accur_all_trials.mean(axis=0)
# axes[2].fill_between(po_vec, mean_val + std_val, mean_val - std_val, alpha = 0.5)
# axes[2].plot(po_vec, mean_val, label = "Accuracy")
# axes[2].legend(loc = "lower right", fontsize = 16)

# axes[2].set_xticks(ticks = np.linspace(po_vec[0], 1.0, 6), fontsize = 16)
# axes[2].set_yticks(ticks = np.linspace(-2.5, 2.5, 3), fontsize = 16)
# axes[2].set_xlim((po_vec[0], po_vec[-2]))

# axes[2].set_xlabel('Likelihood parameter: $p_{\mathcal{O}}$', fontsize = 18)
# axes[1].set_ylabel('$\hat{\mathcal{F}}$ (variational free energy)', fontsize = 18)
# axes[0].set_title('Average $\mathcal{F}$ (and components) as a function of $p_{\mathcal{O}}$', fontsize = 20)

plt.savefig('averageVFE_withcomponents_as_pO.png', dpi = 325)

#%% Run a sweep over values of po (smaller range than above) and measure average variational free energy
ps = 0.5

N, T = 500, 1000

po_vec = np.linspace(0.5, 0.7, 50)

n_trials = 10

vfe_all_trials = np.zeros( (n_trials, len(po_vec) ) )
kld_all_trials = np.zeros( (n_trials, len(po_vec) ) )
accur_all_trials = np.zeros( (n_trials, len(po_vec) ) )

for (ii, po) in enumerate(po_vec):

    for trial_i in range(n_trials):
        G = nx.fast_gnp_random_graph(N, 0.01)
        A = nx.to_numpy_array(G)

        initial_posteriors = np.random.rand(N)
        initial_spins = (initial_posteriors > 0.5).astype(float)
        phi_hist, spin_hist, kld_hist, accur_hist = run_simulation(T, initial_spins, initial_posteriors, A, po, ps)

        vfe_hist = kld_hist - accur_hist
    
        vfe_all_trials[trial_i,ii] = vfe_hist.mean()
        kld_all_trials[trial_i,ii] = kld_hist.mean()
        accur_all_trials[trial_i,ii] = accur_hist.mean()

#%% plot average value of VFE as a function of po (smaller range than above) , with confidence intervals

fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (10, 6))

std_val = 1.96 * vfe_all_trials.std(axis=0)
mean_val = vfe_all_trials.mean(axis=0)
axes[0].fill_between(po_vec, mean_val + std_val, mean_val - std_val, alpha = 0.5)
axes[0].plot(po_vec, mean_val, label = "$\mathcal{F}$")
axes[0].legend(loc = "lower right", fontsize=16)

axes[0].set_xticks(ticks = np.linspace(po_vec[0], 1.0, 6), fontsize = 16)
axes[0].set_yticks(ticks = np.linspace(-2.5, 5.0, 4), fontsize = 16)
axes[0].set_xlim((po_vec[0], po_vec[-2]))

std_val = 1.96 * kld_all_trials.std(axis=0)
mean_val = kld_all_trials.mean(axis=0)
axes[1].fill_between(po_vec, mean_val + std_val, mean_val - std_val, alpha = 0.5)
axes[1].plot(po_vec, mean_val, label = "Complexity")
axes[1].legend(loc = "lower right", fontsize=16)

axes[1].set_xticks(ticks = np.linspace(po_vec[0], 1.0, 6), fontsize = 16)
axes[1].set_yticks(ticks = np.linspace(-3., 3., 4), fontsize = 16)
axes[1].set_xlim((po_vec[0], po_vec[-2]))

std_val = 1.96 * accur_all_trials.std(axis=0)
mean_val = -accur_all_trials.mean(axis=0)
axes[1].fill_between(po_vec, mean_val + std_val, mean_val - std_val, alpha = 0.5)
axes[1].plot(po_vec, mean_val, label = "(-ve) Accuracy")
axes[1].legend(loc = "lower right", fontsize=16)

axes[1].set_xlabel('Likelihood parameter: $p_{\mathcal{O}}$', fontsize = 18)
axes[1].set_ylabel('Average nats', fontsize = 16)

axes[0].set_ylabel('$\hat{\mathcal{F}}$ (variational free energy)', fontsize = 16)

plt.savefig('averageVFE_withcomponents_as_pO_smaller_range.png', dpi = 325)


#%% Run a sweep over values of po and measure criticality using MR estimator
ps = 0.5

T = 1000

po_vec = np.linspace(0.5, 0.999, 100)

n_trials = 100

mre_all_trials = np.zeros( (n_trials, len(po_vec) ) )

for (ii, po) in enumerate(po_vec):

    for trial_i in range(n_trials):
        initial_posteriors = np.random.rand(N)
        initial_spins = (initial_posteriors > 0.5).astype(float)
        phi_hist, spin_hist = run_simulation(T, initial_spins, initial_posteriors, A, po, ps)

        #exclude transient at the beginning of the run
        average_activity = phi_hist[:,100:].mean(axis=0)
        A_next = average_activity[1:]
        A_past = average_activity[:-1]

        A_past_centered = (A_past - A_past.mean())
        numerator = (A_next - A_next.mean()).T @ A_past_centered
        denominator = (A_past_centered**2).sum()

        mre_all_trials[trial_i,ii] = numerator/denominator

#%% plot average value of m as a function of po, with confidence intervals

std_val = 1.96 * mre_all_trials.std(axis=0)
mean_val = mre_all_trials.mean(axis=0)

plt.figure(figsize = (10, 6))
plt.fill_between(po_vec, mean_val + std_val, mean_val - std_val, alpha = 0.5)
plt.plot(po_vec, mean_val)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim((po_vec[0], po_vec[-2]))

plt.xlabel('Likelihood parameter: $p_{\mathcal{O}}$', fontsize = 18)
plt.ylabel('$\hat{m}$ (branching parameter)', fontsize = 18)
plt.title('Branching parameter as a function of $p_{\mathcal{O}}$', fontsize = 20)

plt.savefig('branching_parameter_as_pO.png', dpi = 325)


# %%
