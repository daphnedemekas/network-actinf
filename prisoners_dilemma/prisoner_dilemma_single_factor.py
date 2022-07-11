from pymdp import utils 
import numpy as np 
from pymdp.agent import Agent
from .utils import construct_A, construct_B
from pymdp import maths
num_observations = (4) #((1, -2, 2, 0) reward levels
num_actions = 2 #cooperate, cheat
num_states = (4)  #((the possible combinations), (procosial, antisocial))
#(cooperate & cooperate): ++, (cooperate & defect): +-, (defect&cooperate): -+ , (defect&defect): --

""" Change to """
num_modalities = 1
num_factors = 1


#first modality
""" The probability of being in reward states cc or dc are more likely if the neighbour is prosocial"""

A = construct_A(precision=2)

print("A1: observation of reward to reward states")
print(A[0])
#print(A[0][:,:,1])
print()

""" How to correctly initialize B is a question still"""
B = utils.obj_array(num_factors)
B_1 = np.zeros((4,4,2))
"""
B_1_c = np.array([[p(cc | cc, c), p(cc | cd, c), p(cc | dc, c), p(cc | dd, c)],

                  [p(cd | cc, c), p(cd | cd, c), p(cd | dc, c), p(cd | dd, c)],

                  [0.0,0.0,0.0,0.0],
                  [0.0,0.0,0.0,0.0]])

B_1_d = np.array([
                  [0.0,0.0,0.0,0.0],
                  [0.0,0.0,0.0,0.0],
                  [p(dc | cc, d), p(dc | cd, d), p(dc | dc, d), p(dc | dd, d)],

                  [p(dd | cc, d), p(dd | cd, d), p(dd | dc, d), p(dd | dd, d)],

                  
                  ])
"""


B_1_c = np.array([[0.9, 0.65, 0.2, 0.4],
                  [0.1, 0.35, 0.8, 0.6],
                  [0.0,0.0,0.0,0.0],
                  [0.0,0.0,0.0,0.0]])

B_1_d = np.array([[0.0,0.0,0.0,0.0],
                  [0.0,0.0,0.0,0.0],
                  [0.6, 0.65, 0.2, 0.1],
                  [0.4, 0.35, 0.8, 0.9]])
B_1[:,:,0] = B_1_c
B_1[:,:,1] = B_1_d
B[0] = B_1

#B = construct_B()
#B_2 = np.zeros((2,2,1))
#B_2[:,:,0] = np.array([[0.6,0.4], #the probability p (s = prosocial | s_t=1 = prosocial)
               #       [0.4,0.6]])

#B[1] = B_2# this is probably uniform, need to add noise

print("B1: transitions from reward states given action cooperate")
print(B[0][:,:,0])
print()
print("B1: transitions from reward states given action cheat")
print(B[0][:,:,1])

""" how actions change their actions(put this back)"""
#print("B2: transitions from sociality states ")
#print(B[1])
#print()
#print("B2: transitions from cooperation states given action cheat")
#print(B[1][:,:,1])

C = utils.obj_array(num_modalities)
C[0] = np.array([3,1,4,2])
lr_pb = 0.25

D = utils.obj_array(num_factors)

D[0] = np.array([0.25,0.25,0.25,0.25])
#D[1] = np.array([0.5, 0.5])

pB_1 = utils.dirichlet_like(B)

pB_2 = utils.dirichlet_like(B)

agent_1 = Agent(A=A, B=B, C=C,D=D, pB = pB_1, lr_pB = 1e2)
agent_2 = Agent(A=A, B=B, C=C,D=D, pB = pB_2, lr_pB = 1e2)

observation_1 = [0]
observation_2 = [0]
actions = ["cooperate", "cheat"]

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
qs_prev_1 = D
qs_prev_2 = D
observation_names1 = ["cc","cd","dc","dd"]
observation_names2 = ["prosocial", "antisocial"]

action_names = ["cooperate","cheat"]

def update_state_likelihood_dirichlet(
    pB, B, actions, qs, qs_prev, lr=1.0, factors="all"
):

    num_factors = len(pB)

    qB = copy.deepcopy(pB)
   
    if factors == "all":
        factors = list(range(num_factors))

    for factor in factors:
        dfdb = maths.spm_cross(qs[factor], qs_prev[factor])
        dfdb *= (B[factor][:, :, int(actions[factor])] > 0).astype("float")
        print("HERE")
        print(factor)
        print(qB[factor][:,:,int(actions[factor])])
        print(lr*dfdb)
        qB[factor][:,:,int(actions[factor])] += (lr*dfdb)

    return qB

import copy
for t in range(10):
    print(f'time = : {t}')
    if t != 0:
        qs_prev_1 = qs_1
        qs_prev_2 = qs_2
    
    qs_1 = agent_1.infer_states(observation_1)
    qs_2 = agent_2.infer_states(observation_2)
    print("observations")
    print(observation_1)
    print(observation_2)
    print("qs")
    ##print()
    print(qs_2)
    print(qs_1)
    print(A[0].shape)

    print("AGENT 1 A")

    print(A[0][int(observation_1[0]),:])
    print("AGENT 2 A")

    print(A[0][int(observation_2[0]),:])

    q_pi_1, efe_1 = agent_1.infer_policies()
    q_pi_2, efe_2 = agent_2.infer_policies()
    print("q_pi")
    print(q_pi_1)
    print(q_pi_2)
    print("AGENT 1 B")
    print(agent_1.B[0][:,int(observation_1[0]),0])
    print(agent_1.B[0][:,int(observation_1[0]),1])
    #print(agent_2.B[0][:,int(observation_1[0]),0])
    #print("AGENT 2 B")
    #print(agent_2.B[0][:,int(observation_1[0]),1])

    #print()

    action_1 = agent_1.sample_action()
    action_2 = agent_2.sample_action()
    #print("full actions")
    #print(action_1)
    #print(action_2)
    action_1 = action_1[0]
    action_2 = action_2[0]
    #print("chosen actions")
    #print(action_1)
    #print(action_2)
    #print()
    
    print(f"focal action: {action_names[int(action_1)]}")
    print(f"neighbour action: {action_names[int(action_2)]}")
    observation_1 = get_observation(action_1, action_2)
    observation_2 = get_observation(action_2, action_1)
   # print("observation")
   # print(observation_1)

    num_factors = len(pB_1)

    #print(agent_1.B)

    qB_1 = agent_1.update_B(qs_prev_1)
    qB_2 = agent_2.update_B(qs_prev_2)

    #print(agent_1.B)

    #qB = update_state_likelihood_dirichlet(
    #agent_1.pB, agent_1.B, agent_1.action, agent_1.qs, qs_prev_1, lr=1.0, factors="all"
    #)
    #print(utils.norm_dist_obj_arr(qB))

    #print(qB_1)

    #print(utils.norm_dist_obj_arr(qB_1))

   # print(qs_1)
  #  print(qs_2)
