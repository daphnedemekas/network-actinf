#%% Imports
import numpy as np
from matplotlib import pyplot as plt

#%% Setting up the omegas and the k's

omega_vec = np.linspace(0.0, 1.0, 20)

k_prec_vec = np.linspace(0.0, 5.0, 10)

# def compute_exponential_term(omega, k):
#     """
#     Quick function for computing the big exponential /softmax term at the end of the update
#     """

#     omega_C = 1. - omega
#     exp_k_omega = np.exp(k * omega)
#     exp_k_omega_C = np.exp(k * omega_C)
#     denom = exp_k_omega + exp_k_omega_C
#     numer = omega * exp_k_omega + omega_C * exp_k_omega_C
#     return - 2 * (numer / denom)


def compute_exponential_term(omega, k):
    """
    Quick function for computing the big exponential /softmax term at the end of the update
    """
    omega_C = 1.0 - omega
    exp_k_omega_special = np.exp(k * (1.0 - 2 * omega))
    numer = omega + omega_C * exp_k_omega_special
    denom = 1.0 + exp_k_omega_special
    return 2 * (numer / denom)


# %%

evals = np.empty(shape=(len(omega_vec), len(k_prec_vec)))

for (ii, omega) in enumerate(omega_vec):
    for (jj, k) in enumerate(k_prec_vec):
        evals[ii, jj] = compute_exponential_term(omega, k)

# %% Plot the term as a function of omega for different settings of k

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
for (jj, k) in enumerate(k_prec_vec):
    ax.plot(omega_vec, evals[:, jj], label=f"k = {k.round(3)}")
plt.legend(loc="lower left", fontsize=12)
# %%
