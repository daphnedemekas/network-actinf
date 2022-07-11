import numpy as np

A = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])


""" The question is to learn the B matrix - i.e. the strategy and to see
if it converges to the known optimal straegies, i.e. the players both maximize their rewards"""

from prisoners_dilemma.functions import * 
from pymdp import utils, maths


""" Need help constructing this B matrix"""
B = np.array([maths.softmax(3*np.eye(4)),maths.softmax(3*np.eye(4))])
B = B.reshape((4,4,2))

B = np.ones((4,4,2))*0.25
B[2:,:,0] = 0.01
B[:2,:,1] = 0.01
B = utils.norm_dist(B)

C = np.array([2,4,1,3])
C = C / C.sum()
lr_pb = 0.25

cooperate = 0
cheat = 1

action_1 = cooperate
action_2 = cooperate
observation_1 = 1
observation_2 = 2


D = np.array([1,0,0,0])

prior_1 = D
prior_2 = D
actions = ["cooperate", "cheat"]
B_1 = B
B_2 = B
pB = utils.dirichlet_like(B, scale = 1e16)[0]
pB_1 = pB_2 = pB
qs_prev_1 = D
qs_prev_2 = D

def get_observation(action_1, action_2):
    if action_1 == 0 and action_2 == 0:
        return 0
    elif action_1 == 0 and action_2 == 1:
        return 1
    elif action_1 == 1 and action_2 == 0:
        return 2
    elif action_1 == 1 and action_2 == 1:
        return 3

for t in range(50):
    print(f'time = : {t}')
    if t != 0:
        qs_prev_1 = qs_1
        qs_prev_2 = qs_2

    print(f'prior_1: {prior_1}')
    print(f'prior_2: {prior_2}')

    qs_1, action_1, prior_1 = agent_loop(observation_1, A, prior_1, B_1, C, actions)


    # update generative process
    action_1_label = actions[action_1]
    action_2_label = actions[action_2]

    print(f'action_1: {action_1_label}')
    print(f'action_2: {action_2_label}')

    qs_2, action_2, prior_2 = agent_loop(observation_2, A, prior_2, B_2, C, actions)


    pB_1, B_1 = update_B(pB_1, B_1, action_1, qs_1, qs_prev_1)
    pB_2, B_2 = update_B(pB_2, B_2, action_2, qs_2, qs_prev_2)

    observation_1 = get_observation(action_1, action_2)
    observation_2 = get_observation(action_2, action_1)
    print(f'observation_1: {observation_1}')
    print(f'observation_2: {observation_2}')






print(B_1.shape)
print(B_1[:,:,0])
print()
print(B_1[:,:,1])
print()
print()
print(B_2.shape)
print(B_2[:,:,0])
print()
print(B_2[:,:,1])