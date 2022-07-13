from random import sample
from pymdp import utils
import numpy as np
from pymdp.agent import Agent
from .utils import *
import copy

num_observations = 4  # ((1, -2, 2, 0) reward levels
# (cooperate & cooperate): ++, (cooperate & defect): +-, (defect&cooperate): -+ , (defect&defect): --

num_actions = 2  # cooperate, cheat
num_states = (2, 2)  #  my cooperation/defection state, and your cooperation/defection state

num_modalities = 1
num_factors = 2

A = utils.obj_array(1)
A_1 = np.zeros((4,2,2))

A_1[0,0,0] = 1
A_1[1,1,0] = 1

A_1[2,0,1] = 1
A_1[3,1,1] = 1
A[0] = A_1
B = utils.obj_array(num_factors)

B_1 = np.zeros((2,2,2))

B_1[:,:,0] = np.array([[1,1],[0,0]])
B_1[:,:,1] = np.array([[0,0],[1,1]])
B[0] = B_1

B_2 = np.zeros((2,2,2))

B_2[:,:,0] = np.array([[0.5,0.5],[0.5,0.5]])
B_2[:,:,1] = np.array([[0.5,0.5],[0.5,0.5]])
B[1] = B_2

print(B)

C = utils.obj_array(num_modalities)
C[0] = np.array([3, 1, 4, 2])
lr_pb = 0.25

D = utils.obj_array(2)

D[0] = np.array([0.5, 0.5])
D[1] = np.array([0.5, 0.5])

pB_1 = utils.dirichlet_like(B)

pB_2 = utils.dirichlet_like(B)

agent_1 = Agent(A=A, B=B, C=C, D=D, pB=pB_1, lr_pB=10, factors_to_learn=[1])
agent_2 = Agent(A=A, B=B, C=C, D=D, pB=pB_2, lr_pB=10, factors_to_learn=[1])

observation_1 = [0]
observation_2 = [0]
actions = ["cooperate", "cheat"]

qs_prev_1 = D
qs_prev_2 = D
observation_names1 = ["cc", "cd", "dc", "dd"]
observation_names2 = ["prosocial", "antisocial"]

action_names = ["cooperate", "cheat"]

T = 40


actions_over_time = np.zeros((T, 2))
B_over_time = np.zeros((T, 2, 4, 2))
q_pi_over_time = np.zeros((T, 2, 2))

for t in range(T):
    print(f"time = : {t}")
    if t != 0:
        qs_prev_1 = qs_1
        qs_prev_2 = qs_2
    agent_1.reset()
    agent_2.reset()
    qs_1 = agent_1.infer_states(observation_1)
    qs_2 = agent_2.infer_states(observation_2)

    print("observations")
    print(observation_1)
    print(observation_2)

    print("qs")
    print(qs_1)
    print(qs_2)
    print()

    q_pi_1, efe_1 = agent_1.infer_policies()
    q_pi_2, efe_2 = agent_2.infer_policies()
    print("q_pi")
    print(q_pi_1)
    print()
    # action_1 = agent_1.sample_action()
    # action_2 = agent_2.sample_action()
    action_1 = sample_action_policy_directly(
        q_pi_1, agent_1.policies, agent_1.num_controls
    )
    action_2 = sample_action_policy_directly(
        q_pi_2, agent_2.policies, agent_2.num_controls
    )
    agent_1.action = action_1
    agent_2.action = action_2

    action_1 = action_1[1]
    action_2 = action_2[1]
    actions_over_time[t] = [action_1, action_2]

    observation_1 = get_observation(action_1, action_2)
    observation_2 = get_observation(action_2, action_1)
    """
    print("AGENT 1 B")
    print(agent_1.B[0][:,int(observation_1[0]),0])
    print(agent_1.B[0][:,int(observation_1[0]),1])
    print("AGENT 2 B")
    print(agent_2.B[0][:,int(observation_2[0]),0])
    print(agent_2.B[0][:,int(observation_2[0]),1])
    """
    num_factors = len(pB_1)

    qB = copy.deepcopy(pB_1)

    qB_1 = agent_1.update_B(qs_prev_1)
    qB_2 = agent_2.update_B(qs_prev_2)

"""
what if there are just two hidden state factors -- my cooperation/defection state, and your cooperation/defection state
so the A matrix is like the original 4 x 4, but factorized into a 4 x 2 x 2 
and then the first B matrix (B[0]) is just controlling whether to defect or cooperate -- this just looks like the standard controllable B matrix:

1 1
0 0

in the first slice B[:,:,0]

0 0
1 1

in the second slice B[:,:,1] 
and the second B matrixc (B[1]) is controlling whether "they" defect or cooperate
which you learn. So for example if you learn a tit-for-tat strategy, it might look like:

1 1
0 0

for B[:,:,0] , and similarly
0 0
1 1

for B[:,:,1] 

This encodes the belief that when I cooperate they cooperate, and when I defect they defect. Different structures could encode the belief that when I cooperate they defect (i.e. I learn that they will take advantage of me), or that when I defect, they cooperate (I learn that they can be taken advantage of) 
and learning how they act as a function of your actions is learning how they respond to your actions, right?
and the A matrix just encodes the rules of the game
the second hidden state factor should then also play the role of the 'sociality/antisociality' hidden state factor, since it's literally just your belief about whether they will defect or cooperate
the probability of which could be interpreted as some sociality score
is there something missing here though? I think a problem with this construction is that it doesn't quite capture tit-for-tat, because of the memory issue. It just encodes how you think they will respond this turn to your simultaneous action, rather than how they will respond in the next turn. But unless I'm misunderstanding something, I think even the current formulation suffers from this memory issue 
"""