import numpy as np
from pymdp.maths import softmax
from pymdp import utils


def construct_A(precision_prosocial=3.0, precision_antisocial=2.0):
    A1_prosocial = np.zeros((4, 4))
    A1_prosocial[:, 0] = softmax(precision_prosocial * np.array([1, 0, 0, 0]))
    A1_prosocial[:, 1] = softmax(precision_antisocial * np.array([0, 1, 0, 0]))
    A1_prosocial[:, 2] = softmax(precision_prosocial * np.array([0, 0, 1, 0]))
    A1_prosocial[:, 3] = softmax(precision_antisocial * np.array([0, 0, 0, 1]))

    A1_antisocial = np.zeros((4, 4))
    A1_antisocial[:, 0] = softmax(
        precision_antisocial * np.array([1, 0, 0, 0])
    )
    A1_antisocial[:, 1] = softmax(precision_prosocial * np.array([0, 1, 0, 0]))
    A1_antisocial[:, 2] = softmax(
        precision_antisocial * np.array([0, 0, 1, 0])
    )
    A1_antisocial[:, 3] = softmax(precision_prosocial * np.array([0, 0, 0, 1]))
    A = utils.obj_array(1)

    A1 = np.zeros((4, 4, 2))
    A1[:, :, 0] = A1_prosocial
    A1[:, :, 1] = A1_antisocial
    A[0] = A1
    return A


def sample_action_policy_directly(q_pi, policies, num_controls, style = "deterministic"):

    num_factors = len(num_controls)

    if style == "deterministic":
        policy_idx = np.argmax(q_pi)
    elif style == "stochastic":
        policy_idx = utils.sample(q_pi)

    selected_policy = np.zeros(num_factors)
    for factor_i in range(num_factors):
        selected_policy[factor_i] = policies[policy_idx][0, factor_i]

    return selected_policy


def get_observation(action_1, action_2):
    action_1 == int(action_1)
    action_2 = int(action_2)
    if action_1 == 0 and action_2 == 0:
        return [0]
    elif action_1 == 0 and action_2 == 1:
        return [1]
    elif action_1 == 1 and action_2 == 0:
        return [2]
    elif action_1 == 1 and action_2 == 1:
        return [3]


def print_A(A):
    print("A1: observation of reward to reward states")
    print(A[0])
    print(A[0].shape)
    print(A[0][3, :, :])

    print()


def print_B(B):
    print("B1: transitions from reward states given action cooperate")
    print(B[0][:, :, 0])
    print()
    print("B1: transitions from reward states given action cheat")
    print(B[0][:, :, 1])

    print("B2: transitions from cooperation states given action cooperate")
    print(B[1][:, :, 0])
    print()
    print("B2: transitions from cooperation states given action cheat")
    print(B[1][:, :, 1])
