import numpy as np
from pymdp.maths import softmax
from pymdp import utils
from pymdp.agent import Agent


num_observations = 4  # ((1, -2, 2, 0) reward levels
num_actions = 2  # cooperate, cheat
num_states = (4, 2)  # ((the possible combinations), (procosial, antisocial))
# (cooperate & cooperate): ++, (cooperate & defect): +-, (defect&cooperate): -+ , (defect&defect): --

num_modalities = 1
num_factors = 2


def construct(precision_prosocial, precision_antisocial, lr_pB,factors_to_learn):
    A = construct_A(precision_prosocial, precision_antisocial)

    # print_A(A)

    B = utils.obj_array(num_factors)
    B_1 = np.ones((4, 4, 2)) * 0.5
    B_1[2:, :, 0] = 0.0
    B_1[:2, :, 1] = 0.0
    B[0] = B_1

    B_2 = np.zeros((2, 2, 2))

    B_2[:, :, 0] = np.array(
        [
            [0.5, 0.5],  # given that i cooperated, the probability p (s = prosocial | s_t=1 = prosocial)
            [0.5, 0.5],
        ] )

    B_2[:, :, 1] = np.array(
        [
            [0.5, 0.5],  # given that i cooperated, the probability p (s = prosocial | s_t=1 = prosocial)
            [0.5, 0.5],
        ] )

    B[1] = B_2


    # print_B(B)

    C = utils.obj_array(num_modalities)
    C[0] = np.array([3, 1, 4, 2])

    D = utils.obj_array(num_factors)

    D[0] = np.array([0.25, 0.25, 0.25, 0.25])
    D[1] = np.array([0.5, 0.5])

    pB_1 = utils.dirichlet_like(B)

    pB_2 = utils.dirichlet_like(B)


    agent_1 = Agent(A=A, B=B, C=C, D=D, pB = pB_1, lr_pB = lr_pB, policies = [np.array([[0,0]]), np.array([[1, 1]])],factors_to_learn = factors_to_learn)
    agent_2 = Agent(A=A, B=B, C=C, D=D, pB = pB_2,  lr_pB = lr_pB, policies = [np.array([[0,0]]), np.array([[1, 1]])],factors_to_learn = factors_to_learn)

    return agent_1, agent_2, D



def construct_2(lr_pB,factors_to_learn):
    A = construct_A_2()
    num_factors = 1
    # print_A(A)

    B = utils.obj_array(num_factors)
    B_1 = np.ones((4, 4, 2)) * 0.5
    B_1[2:, :, 0] = 0.0
    B_1[:2, :, 1] = 0.0
    B[0] = B_1

    # print_B(B)

    C = utils.obj_array(num_modalities)
    C[0] = np.array([3, 1, 4, 2])

    D = utils.obj_array(num_factors)

    D[0] = np.array([0.25, 0.25, 0.25, 0.25])

    pB_1 = utils.dirichlet_like(B)

    pB_2 = utils.dirichlet_like(B)


    agent_1 = Agent(A=A, B=B, C=C, D=D, pB = pB_1, lr_pB = lr_pB, factors_to_learn = factors_to_learn)
    agent_2 = Agent(A=A, B=B, C=C, D=D, pB = pB_2,  lr_pB = lr_pB, factors_to_learn = factors_to_learn)

    return agent_1, agent_2, D


def sweep(agent_1, agent_2, observation_1, observation_2, D, T, sample_style = 'deterministic'):
    # first modality
    qs_prev_1 = D
    qs_prev_2 = D

    actions_over_time = np.zeros((T, 2))
    B1_over_time = np.zeros((T, 4, 4, 2, 2))
    B2_over_time = np.zeros((T, 2, 2, 2, 2))

    q_pi_over_time = np.zeros((T, 2, 2))


    for t in range(T):
        qs_1 = agent_1.infer_states(observation_1)
        qs_2 = agent_2.infer_states(observation_2)
        if t > 0:
            qB_1 = agent_1.update_B(qs_prev_1)
            qB_2 = agent_2.update_B(qs_prev_2)

        q_pi_1, efe_1 = agent_1.infer_policies()
        q_pi_2, efe_2 = agent_2.infer_policies()
        q_pi_over_time[t,:,0] = q_pi_1
        q_pi_over_time[t,:,1] = q_pi_2

        action_1 = sample_action_policy_directly(
            q_pi_1, agent_1.policies, agent_1.num_controls, style=sample_style
        )
        action_2 = sample_action_policy_directly(
            q_pi_2, agent_2.policies, agent_2.num_controls, style=sample_style
        )
        agent_1.action = action_1
        agent_2.action = action_2

        qs_prev_1 = qs_1
        qs_prev_2 = qs_2


        action_1 = action_1[1]
        action_2 = action_2[1]

        observation_1 = get_observation(action_1, action_2)
        observation_2 = get_observation(action_2, action_1)

        actions_over_time[t] = [action_1, action_2]

        B1_over_time[t, :,:,:, 0] = agent_1.B[0]
        B1_over_time[t,:,:,:, 0] = agent_1.B[0]
        B1_over_time[t, :,:,:, 1] = agent_2.B[0]
        B1_over_time[t,:,:,:, 1] = agent_2.B[0]

        B2_over_time[t, :,:,:, 0] = agent_1.B[1]
        B2_over_time[t,:,:,:, 0] = agent_1.B[1]
        B2_over_time[t, :,:,:, 1] = agent_2.B[1]
        B2_over_time[t,:,:,:, 1] = agent_2.B[1]
    return actions_over_time, B1_over_time, B2_over_time, q_pi_over_time


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

def sweep_2(agent_1, agent_2, observation_1, observation_2, D, T, sample_style = 'deterministic'):
    # first modality
    qs_prev_1 = D
    qs_prev_2 = D

    actions_over_time = np.zeros((T, 2))
    B1_over_time = np.zeros((T, 4, 4, 2, 2))

    q_pi_over_time = np.zeros((T, 2, 2))


    for t in range(T):
        qs_1 = agent_1.infer_states(observation_1)
        qs_2 = agent_2.infer_states(observation_2)
        if t > 0:
            qB_1 = agent_1.update_B(qs_prev_1)
            qB_2 = agent_2.update_B(qs_prev_2)

        q_pi_1, efe_1 = agent_1.infer_policies()
        q_pi_2, efe_2 = agent_2.infer_policies()
        q_pi_over_time[t,:,0] = q_pi_1
        q_pi_over_time[t,:,1] = q_pi_2

        action_1 = agent_1.sample_action(sample_style = sample_style)
        action_2 = agent_2.sample_action()
        agent_1.action = action_1
        agent_2.action = action_2

        qs_prev_1 = qs_1
        qs_prev_2 = qs_2

        action_1 = action_1[0]
        action_2 = action_2[0]

        observation_1 = get_observation(action_1, action_2)
        observation_2 = get_observation(action_2, action_1)

        actions_over_time[t] = [action_1, action_2]

        B1_over_time[t, :,:,:, 0] = agent_1.B[0]
        B1_over_time[t,:,:,:, 0] = agent_1.B[0]
        B1_over_time[t, :,:,:, 1] = agent_2.B[0]
        B1_over_time[t,:,:,:, 1] = agent_2.B[0]
    return actions_over_time, B1_over_time, q_pi_over_time

def construct_A_2():
    A = utils.obj_array(1)
    A1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
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
