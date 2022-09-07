#%%

from random import sample
from pymdp import utils
import numpy as np
from pymdp.agent import Agent
from utils import *
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

T = 1000
num_trials = 1

actions_over_time_all = np.zeros((T, 1,100,100,num_trials))
#B1_over_time_all = np.zeros((T, 4, 4, 2, 2,30,30,num_trials))

#q_pi_over_time_all = np.zeros((T, 2, 2,30,30,num_trials))

for k, lr_pB_1 in enumerate(np.linspace(0.01,0.6,100)):
    print(f"lr = : {lr_pB_1}")
    for j, lr_pB_2 in enumerate(np.linspace(0.01,0.6,100)):
        for t in range(num_trials):

            #alpha_1 = np.random.normal(alpha_m, 0.15)
            #alpha_2 = np.random.normal(alpha_m, 0.15)

            agent_1, agent_2, D = construct_2(lr_pB = lr_pB_1,lr_pB_2 = lr_pB_2,factors_to_learn="all")
            #agent_1.action_selection = "stochastic"
            #agent_2.action_selection = "stochastic"
            #agent_1.alpha = alpha_1
            #agent_2.alpha = alpha_2
            actions_over_time = sweep_3(agent_1, agent_2, observation_1 = [0], observation_2 = [0],D=D,T=T)

            #B1_over_time_all[:,:,:,:,:,k,j,t] = B1_over_time
            #q_pi_over_time_all[:,:,:,k,j,t] = q_pi_over_time
            actions_over_time_all[:,:,k,j,t] = actions_over_time

np.save('actions_over_time_all',actions_over_time_all,allow_pickle = True)
#np.save('B1_over_time_all',B1_over_time_all,allow_pickle = True)
#np.save('q_pi_over_time_all',q_pi_over_time_all,allow_pickle = True)