#%%

from random import sample
from pymdp import utils
import numpy as np
from pymdp.agent import Agent
from .utils import *
import copy

num_observations = 4  # ((1, -2, 2, 0) reward levels
num_actions = 2  # cooperate, cheat
num_states = (4, 2)  # ((the possible combinations), (procosial, antisocial))
# (cooperate & cooperate): ++, (cooperate & defect): +-, (defect&cooperate): -+ , (defect&defect): --

num_modalities = 1
num_factors = 2
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
def construct(precision_prosocial, precision_antisocial, lr_pB):
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


    agent_1 = Agent(A=A, B=B, C=C, D=D, pB = pB_1, lr_pB = lr_pB, policies = [np.array([[0,0]]), np.array([[1, 1]])])
    agent_2 = Agent(A=A, B=B, C=C, D=D, pB = pB_2,  lr_pB = lr_pB, policies = [np.array([[0,0]]), np.array([[1, 1]])])

    return agent_1, agent_2, D
def sweep(agent_1, agent_2, D):
# first modality
    observation_1 = [np.random.choice([0,1])]
    observation_2 = [np.random.choice([0,1])]
    actions = ["cooperate", "cheat"]

    qs_prev_1 = D
    qs_prev_2 = D
    observation_names1 = ["cc", "cd", "dc", "dd"]
    observation_names2 = ["prosocial", "antisocial"]

    action_names = ["cooperate", "cheat"]

    T = 300


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
            q_pi_1, agent_1.policies, agent_1.num_controls, style='deterministic'
        )
        action_2 = sample_action_policy_directly(
            q_pi_2, agent_2.policies, agent_2.num_controls, style='deterministic'
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

T = 300

actions_over_time_all = np.zeros((T, 2, 1,1,5,100))
B1_over_time_all = np.zeros((T, 4, 4, 2, 2, 1,1,5,100))
B2_over_time_all = np.zeros((T, 2, 2, 2, 2, 1,1,5,100))

q_pi_over_time_all = np.zeros((T, 2, 2, 1,1,5,100))
num_trials = 100

for i, p_m in enumerate([4.0]):
    print(f"p = : {p_m}")

    for j, pa_m in enumerate([5]):
        print(f"pa = : {pa_m}")

        for k, lr_m in enumerate([0.0,0.3,0.5,0.8,1.0]):
            print(f"lr = : {lr_m}")

            for t in range(num_trials):
                p = np.random.normal(p_m, 0.05)
                pa = np.random.normal(pa_m, 0.05)
                lr_pB = np.random.normal(lr_m, 0.05)
                if lr_pB < 0:
                    lr_pB = 0
                agent_1, agent_2, D = construct(precision_prosocial = p, precision_antisocial = pa,lr_pB = lr_pB)

                actions_over_time, B1_over_time, B2_over_time, q_pi_over_time = sweep(agent_1, agent_2, D)
                actions_over_time_all[:,:,i,j,k,t] = actions_over_time

                B1_over_time_all[:,:,:,:,:,i,j,k,t] = B1_over_time
                B2_over_time_all[:,:,:,:,:,i,j,k,t] = B2_over_time
                q_pi_over_time_all[:,:,:,i,j,k,t] = q_pi_over_time

np.save('actions_over_time_all',actions_over_time_all,allow_pickle = True)
np.save('B1_over_time_all',B1_over_time_all,allow_pickle = True)
np.save('B2_over_time_all',B2_over_time_all,allow_pickle = True)
np.save('q_pi_over_time_all',q_pi_over_time_all,allow_pickle = True)