import networkx as nx 
import numpy as np 
from networkx import Graph
from typing import Tuple
from numpy import array 
import matplotlib.pyplot as plt 

from math_utils import log_stable

class Simulation:

    def __init__(self, G: Graph, initial_posteriors = None, initial_spins = None):
        self.network = G

        self.A = nx.to_numpy_array(G)
        self.N = G.number_of_nodes()
        if not initial_posteriors:
            initial_posteriors = np.random.rand(self.N) # posterior is belief about belign in down state
        self.initial_posteriors = initial_posteriors
        if not initial_spins:
            initial_spins = (self.initial_posteriors > 0.5).astype(float) # if spin == 1, you're in a DOWN spin, otherwise, you're in an UP sstate
        self.initial_spins = initial_spins

    def get_log_precision(self, po : float, ps: float) -> Tuple[float, float, float, float]:
        "calculate these in advance for quick use in VFE calculations"
        self.logpo = log_stable(po)
        self.logps = log_stable(ps)
        self.logpo_C = log_stable(1.-po)
        self.logps_C = log_stable(1.-ps)

        return self.logpo, self.logps, self.logpo_C, self.logps_C

    # def get_hist_array(self, T: int) -> Tuple[array, array, array, array, array, array]:
    def get_hist_array(self, T: int) -> Tuple[array, array]:

        spin_hist = np.empty((self.N, T) )
        phi_hist = np.empty((self.N, T) )
        return spin_hist, phi_hist

        # hist = np.zeros((self.N, T) )
        # return hist, hist, hist, hist, hist, hist

    def calculate_spins(self, spin_state: float) -> Tuple[array, array, array]:
        sum_down_spins = self.A @ spin_state # this sums up the neighbours that are DOWN, per agent
        up_spins = np.absolute(spin_state - 1) # converts from 1.0 meaning DOWN to 1.0 meaning UP
        sum_up_spins = self.A @ up_spins # this sums up the neighbours that are UP, per agent
        spin_diffs = sum_down_spins - sum_up_spins # difference in DOWNs vs UPs, per agent

        return sum_down_spins, sum_up_spins, spin_diffs

    def sample_spin_state(self, ps: float, po: float, spin_diffs: array) -> Tuple[float, float, float]:
        # equivalent expressions
        # x = ((ps - 1.) * (((1./po) - 1.)**spin_diffs)) / ps
        x = (1. - (1./ps)) * ((1. - po)/po)**(spin_diffs)
        phi = 1. / (1. - x)
        spin_state = (phi > np.random.rand(self.N)).astype(float)
        return phi, 1. - phi, spin_state

    def decompose_neg_entropy_eve(self, logpo: float, logps: float, logpo_C: float, logps_C: float, phi: float, phi_C: float, sum_down_spins: array, sum_up_spins: array) -> Tuple[float, float]:
        negH = phi * np.log(phi + 1e-16) + phi_C * np.log(phi_C + 1e-16)
        neg_expected_energy = phi * (sum_down_spins * logpo + sum_up_spins*logpo_C + logps) + phi_C*(sum_up_spins*logpo + sum_down_spins*logpo_C + logps_C)

        return negH, neg_expected_energy

    def decompose_complexity_accuracy(self, logpo: float, logps: float, logpo_C: float, logps_C: float, phi: float, phi_C: float, sum_down_spins: array, sum_up_spins: array) -> Tuple[float, float]:
        kld = phi * (np.log(phi + 1e-16) - logps) + phi_C * (np.log(phi_C + 1e-16) - logps_C)
        accur = phi * (sum_down_spins*logpo + sum_up_spins*logpo_C) + phi_C*(sum_up_spins*logpo + sum_down_spins*logpo_C)

        return kld, accur

    # def run(self, T: int, po: float, ps: float) -> Tuple[array, array, array, array, array, array]:
    def run(self, T: int, po: float, ps: float) -> Tuple[array, array]:

        # spin states -- 1.0 == DOWN, 0.0 == UP
        spin_state = self.initial_spins.copy()

        # posteriors
        phi = self.initial_posteriors.copy()
        # spin_hist, phi_hist, kld_hist, accur_hist, negH_hist, energy_hist = self.get_hist_array(T)
        spin_hist, phi_hist = self.get_hist_array(T)

        log_precisions = self.get_log_precision(po, ps)

        for t in range(T):

            sum_down_spins, sum_up_spins, spin_diffs = self.calculate_spins(spin_state)

            # sample new spin-state -- probability that I'm DOWN
            phi, phi_C, spin_state = self.sample_spin_state(ps, po, spin_diffs)

            # store histories of spin states and posteriors
            spin_hist[:,t] = spin_state.copy()
            phi_hist[:,t] = phi.copy()

            # compute VFE for each agent (@NOTE: this could be computed outside this function, after the fact - probably should be done in order to speed things up)

            # # Decomposition 1: negative entropy - expected variational energy (uncomment below if you want to do this)
            # negH, neg_expected_energy = self.decompose_neg_entropy_eve(*log_precisions, phi, phi_C, sum_down_spins, sum_up_spins)
            # negH_hist[:,t] = negH
            # energy_hist[:,t] = -neg_expected_energy

            # # Decomposition 2: complexity - accuracy (uncomment below if you want to do this)
            # kld, accur = self.decompose_complexity_accuracy(*log_precisions, phi, phi_C, sum_down_spins, sum_up_spins)

            # kld_hist[:,t] = kld
            # accur_hist[:,t] = accur

        # return phi_hist, spin_hist, kld_hist, accur_hist, negH_hist, energy_hist
        return phi_hist, spin_hist


    def compute_VFE(self, phi_hist: array, spin_hist: array, decomposition: str = "entropy_energy") -> Tuple[array,array, array]:
        """ 
        Compute variational free energy for each agent and timepoint using an input history of beliefs and spins, and parameters of generative models
        """
        
        phi_C_hist = 1. - phi_hist

        down_hist, up_hist = spin_hist, np.absolute(spin_hist - 1.)
        sum_down_spins_hist = self.A @ down_hist
        sum_up_spins_hist = self.A @ up_hist

        if decomposition == "entropy_energy":

            negH = phi_hist * log_stable(phi_hist) + phi_C_hist * log_stable(phi_C_hist)
            expected_energy = - (phi_hist * (sum_down_spins_hist * self.logpo + sum_up_spins_hist*self.logpo_C + self.logps) + phi_C_hist*(sum_up_spins_hist*self.logpo + sum_down_spins_hist*self.logpo_C + self.logps_C))

            vfe = negH + expected_energy
            return vfe, negH, expected_energy

        if decomposition == "complexity_accuracy":

            complexity = phi_hist * (log_stable(phi_hist) - self.logps) + phi_C_hist * (log_stable(phi_C_hist) - self.logps_C)
            neg_accur = -(phi_hist * (sum_down_spins_hist*self.logpo + sum_up_spins_hist*self.logpo_C) + phi_C_hist*(sum_up_spins_hist*self.logpo + sum_down_spins_hist*self.logpo_C))

            vfe = complexity + neg_accur
            return vfe, complexity, neg_accur

    def calculate_average_metric(self, hist: array) -> array:
        return hist.mean(axis = 0)

    def get_regime_data(self, T: int, hist: array):
        A_t = self.calculate_average_metric(hist)

        data = np.arange(T), A_t
        return data


def plot_regimes(regimes: list, po_vec = None, ps_vec = None):

    fig, axes = plt.subplots(nrows = len(regimes), ncols = 1, figsize = (10, 8), sharex = True, sharey = False)
    for i in range(len(regimes)):
        axes[i].plot(regimes[i][0], regimes[i][1])
        axes[i].set_xlim(0, len(regimes[i][0]))
        axes[i].tick_params(axis='both', which='major', labelsize=16)
        if len(po_vec) == len(regimes):
            axes[i].set_title('$p_{\mathcal{O}} = $' + str(po_vec[i].round(2)), fontsize = 18)
        elif len(ps_vec) == len(regimes):
            axes[i].set_title('$p_{\mathcal{O}} = $' + str(po_vec[i].round(2)), fontsize = 18)
        elif len(po_vec) == len(regimes) and len(ps_vec) == len(regimes):
            title = '$p_{\mathcal{O}} = $' + str(po_vec[i].round(2))
            title += ' $p_{\mathcal{O}} = $' + str(po_vec[i].round(2))
            axes[i].set_title(title, fontsize = 12)

    return axes