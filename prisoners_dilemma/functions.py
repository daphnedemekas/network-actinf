import pymdp
from pymdp.learning import update_state_likelihood_dirichlet
from pymdp import utils, control, inference
from pymdp import maths
from pymdp.maths import softmax
from pymdp.maths import spm_log_single as log_stable
import copy
import numpy as np


def update_B(pB, B, action, qs, qs_prev, lr=0.25):
    qB = copy.deepcopy(pB)
    dfdb = maths.spm_cross(qs, qs_prev)
    dfdb *= (B[:, :, int(action)] > 0).astype("float")
    qB[:, :, int(action)] += lr * dfdb
    pB = qB
    B = utils.norm_dist(qB)

    return pB, B


def infer_states(observation, A, prior):

    """Implement inference here -- NOTE: prior is already passed in, so you don't need to do anything with the B matrix. We already have our P(s_t). The conditional expectation should happen _outside_ this function"""

    log_likelihood = log_stable(A[observation, :])
    log_prior = log_stable(prior)

    qs = softmax(log_likelihood + log_prior)

    return qs


def get_expected_states(B, current_qs, action):

    qs_u = B[:, :, action].dot(current_qs)

    return qs_u


def get_expected_observations(A, qs_u):

    qo_u = A.dot(qs_u)

    return qo_u


def entropy(A):

    entropy_A = (-A * log_stable(A)).sum(axis=0)

    return entropy_A


def kl_divergence(qo_u, C):

    kld = (qo_u * (log_stable(qo_u) - log_stable(C))).sum()

    return kld


def calculate_G(A, B, C, qs_current, actions):

    G = np.zeros(len(actions))

    for action_id in range(len(actions)):
        qs_u = get_expected_states(B, qs_current, action_id)
        print(f"expected states: {qs_u}")
        qo_u = get_expected_observations(A, qs_u)
        print(f"expected observations: {qs_u}")

        H_A = entropy(A)

        kld = kl_divergence(qo_u, C)

        G[action_id] = (
            H_A.dot(qs_u) + kld
        )  # predicted uncertainty + predicted divergence

    return G


def agent_loop(observation, A, prior, B, C, actions):
    observation = observation
    qs = infer_states(observation, A, prior)
    print(f"qs: {qs}")
    G = calculate_G(A, B, C, qs, actions)
    Q_u = softmax(-G)
    print(f"Q_u : {Q_u}")
    action = utils.sample(Q_u)
    print(f"action: {action}")

    prior = B[:, :, action].dot(qs)
    return qs, action, prior
