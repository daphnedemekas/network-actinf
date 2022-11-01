import numpy as np
from pymdp.maths import softmax
from pymdp import utils
from pymdp.agent import Agent
import matplotlib.pyplot as plt


num_observations = 4  # ((1, -2, 2, 0) reward levels
num_actions = 2  # cooperate, cheat
# (cooperate & cooperate): ++, (cooperate & defect): +-, (defect&cooperate): -+ , (defect&defect): --

num_modalities = 1
num_factors = 1
from matplotlib.colors import LinearSegmentedColormap
cmap0 = LinearSegmentedColormap.from_list("", ["white", "black"])



def construct_A():
    A = utils.obj_array(1)
    A1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    A[0] = A1
    return A


def construct_B():
    B = utils.obj_array(num_factors)
    B_1 = np.ones((4, 4, 2)) * 0.5
    B_1[2:, :, 0] = 0.0
    B_1[:2, :, 1] = 0.0
    B[0] = B_1

    return B


def get_agent_params():
    """Returns standard parameters for a prisoners dilemma agent"""
    A = construct_A()
    # print_A(A)
    B = construct_B()
    # print_B(B)

    C = utils.obj_array(num_modalities)
    C[0] = np.array([3, 1, 4, 2])

    D = utils.obj_array(num_factors)

    D[0] = np.array([0.25, 0.25, 0.25, 0.25])

    pB_1 = utils.dirichlet_like(B)

    pB_2 = utils.dirichlet_like(B)

    return A, B, C, D, pB_1, pB_2


def construct(lr_pB, lr_pB_2=None, factors_to_learn=None):
    """Constructs two agents for the dual-agent simulation"""
    A, B, C, D, pB_1, pB_2 = get_agent_params()

    agent_1 = Agent(
        A=A, B=B, C=C, D=D, pB=pB_1, lr_pB=lr_pB, factors_to_learn=factors_to_learn
    )
    agent_2 = Agent(
        A=A, B=B, C=C, D=D, pB=pB_2, lr_pB=lr_pB_2, factors_to_learn=factors_to_learn
    )

    return agent_1, agent_2, D


def run_sim_collect_all_data(agent_1, agent_2, observation_1, observation_2, D, T):
    """Here we run a dual-agent simulation and collect actions, transition matrices,
    posterior over states and posterior over policies"""
    # first modality
    qs_prev_1 = D
    qs_prev_2 = D


    actions_over_time = np.zeros((T, 2))
    B1_over_time = np.zeros((T, 4, 4, 2, 2))

    q_pi_over_time = np.zeros((T, 2, 2))
    q_s_over_time = np.zeros((T, 4, 2))

    for t in range(T):
        qs_1 = agent_1.infer_states(observation_1)
        qs_2 = agent_2.infer_states(observation_2)
        q_s_over_time[t, :, 0] = qs_1[0]
        q_s_over_time[t, :, 1] = qs_2[0]
        if t > 0:
            qB_1 = agent_1.update_B(qs_prev_1)
            qB_2 = agent_2.update_B(qs_prev_2)

        q_pi_1, efe_1 = agent_1.infer_policies()
        q_pi_2, efe_2 = agent_2.infer_policies()
        q_pi_over_time[t, :, 0] = q_pi_1
        q_pi_over_time[t, :, 1] = q_pi_2

        action_1 = sample_action(agent_1)
        action_2 = sample_action(agent_2)
        agent_1.action = action_1
        agent_2.action = action_2
        agent_1.step_time()
        agent_2.step_time()

        qs_prev_1 = qs_1
        qs_prev_2 = qs_2

        action_1 = action_1[0]
        action_2 = action_2[0]
        observation_1 = get_observation(action_1, action_2)
        observation_2 = get_observation(action_2, action_1)

        actions_over_time[t] = [action_1, action_2]

        B1_over_time[t, :, :, :, 0] = agent_1.B[0]
        B1_over_time[t, :, :, :, 0] = agent_1.B[0]
        B1_over_time[t, :, :, :, 1] = agent_2.B[0]
        B1_over_time[t, :, :, :, 1] = agent_2.B[0]
    return actions_over_time, B1_over_time, q_pi_over_time, q_s_over_time, agent_1

#strategy
def plot_b_matrices(agent):
    print(agent.B[0][2:,:,1])
    fig = plt.figure(figsize = (4,2))
    plt.xticks(range(4), labels=['CC','CD','DC','DD'],fontsize =14)
    plt.yticks(range(2), labels=['DC','DD'],fontsize =14)
    plt.title("Transitions given action defect")
    plt.imshow(agent.B[0][2:,:,1], cmap=cmap0, vmin = 0, vmax =1)
    plt.show()
    print(agent.B[0][:2,:,0])

    fig = plt.figure(figsize = (4,2))
    plt.xticks(range(4), labels=['CC','CD','DC','DD'],fontsize =14)
    plt.yticks(range(2), labels=['CC','CD'],fontsize =14)
    plt.title("Transitions given action cooperate")
    plt.imshow(agent.B[0][:2,:,0], cmap=cmap0, vmin = 0, vmax =1)



def plot_testing(actions_over_time, aspect):
    T = actions_over_time.shape[0]

    fig, ax = plt.subplots(figsize = (16,15))
    ax.grid(color='w', which = 'minor', linestyle='-', linewidth=2)
    ax.set_xlabel("Time")
    im = ax.imshow(actions_over_time.T,cmap ='gray', aspect = aspect, vmin=0, vmax =1, interpolation="None")
    ax.set_xlabel("Time",fontsize = 12)
    ax.set_xticks(np.linspace(0,T,25).round(0),fontsize = 12)
    ax.set_yticks([0, 1], labels=["Tester", "Agent"],fontsize = 12)
    plt.savefig('testing')
    plt.show()

def sample_action(agent):
    num_factors = len(agent.num_controls)

    action_marginals = utils.obj_array_zeros(agent.num_controls)

    for pol_idx, policy in enumerate(agent.policies):
        for factor_i, action_i in enumerate(policy[0, :]):
            action_marginals[factor_i][action_i] += agent.q_pi[pol_idx]
    
    action_marginals = utils.norm_dist_obj_arr(action_marginals)

    selected_policy = np.zeros(num_factors)
    for factor_i in range(num_factors):

        # Either you do this:
        if agent.action_selection == 'deterministic':
            selected_policy[factor_i] = np.argmax(action_marginals[factor_i])
        elif agent.action_selection == 'stochastic':
            p_actions = softmax(action_marginals[factor_i] * agent.alpha)
            selected_policy[factor_i] = np.array([np.random.choice([0,1],p=p_actions)])
    return selected_policy
def pd_one_round(agent_1, agent_2, observation_1, observation_2, t):
    """Here we run a dual-agent simulation and collect actions, transition matrices,
    posterior over states and posterior over policies"""
    # first modality

    qs_prev_1 = agent_1.D
    qs_prev_2 = agent_1.D

    qs_1 = agent_1.infer_states(observation_1)
    qs_2 = agent_2.infer_states(observation_2)

    if t > 0:
        qB_1 = agent_1.update_B(qs_prev_1)
        qB_2 = agent_2.update_B(qs_prev_2)

    q_pi_1, efe_1 = agent_1.infer_policies()
    q_pi_2, efe_2 = agent_2.infer_policies()

    action_1 = agent_1.sample_action()
    action_2 = agent_2.sample_action()
    agent_1.action = action_1
    agent_2.action = action_2

    observation_1 = get_observation(action_1, action_2)
    observation_2 = get_observation(action_2, action_1)

    agent_1.observation = observation_1
    agent_2.observation = observation_2
    qs_1 = agent_1.infer_states(observation_1)
    qs_2 = agent_2.infer_states(observation_2)
    qB_1 = agent_1.update_B(qs_1)
    qB_2 = agent_2.update_B(qs_2)

    q_pi_1, efe_1 = agent_1.infer_policies()
    q_pi_2, efe_2 = agent_2.infer_policies()

    action_1 = agent_1.sample_action()
    action_2 = agent_2.sample_action()
    agent_1.action = action_1
    agent_2.action = action_2

    return action_1[0], action_2[0], agent_1.B[0], agent_2.B[0], q_pi_1, q_pi_2, qs_1, qs_2


def sweep_with_testing(
    agent_1,
    agent_2,
    observation_1,
    T,
):
    agent_1.lr = 0
    agent_2.lr = 0
    """Sweep function that includes period between test_start_time and test_end_time to input fake observations of defer"""
    # first modality

    actions_over_time = np.zeros((T, 2))

    for t in range(T):
        qs_1 = agent_1.infer_states(observation_1)

        q_pi_1, efe_1 = agent_1.infer_policies()

        action_1 = agent_1.sample_action()

        if np.random.choice([0,1]) == 0:
            action_2 = np.array([1])
        else:
            action_2 = np.array([0])

        agent_1.action = action_1
        action_1 = action_1[0]
        action_2 = action_2[0]
        observation_1 = get_observation(action_1, action_2)

        actions_over_time[t] = [action_2, action_1]

    return actions_over_time, agent_1

def run_sim_collect_actions(agent_1, agent_2, observation_1, observation_2, D, T):
    """Here we run a simulation and return the actions over time for two agents"""
    # first modality
    qs_prev_1 = D
    qs_prev_2 = D

    actions_over_time = np.zeros((T, 2))

    for t in range(T):
        qs_1 = agent_1.infer_states(observation_1)
        qs_2 = agent_2.infer_states(observation_2)

        if t > 0:
            agent_1.update_B(qs_prev_1)
            agent_2.update_B(qs_prev_2)

        agent_1.infer_policies()
        agent_2.infer_policies()

        action_1 = agent_1.sample_action()
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

    return actions_over_time


def sample_action_policy_directly(q_pi, policies, num_controls, style="deterministic"):

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

def plot_sim(actions_over_time, aspect = 10):
    fig, ax = plt.subplots(figsize = (16,15))
    ax.grid(color='w', which = 'minor', linestyle='-', linewidth=2)
    ax.set_xlabel("Time")
    im = ax.imshow(actions_over_time.T,cmap ='gray', aspect = aspect, vmin=0, vmax =1)
    ax.set_xlabel("Time",fontsize = 12)
    ax.set_xticks(np.linspace(0,actions_over_time.shape[0]-1,25).round(0),fontsize = 12)
    ax.set_yticks([0, 1], labels=["Agent 1", "Agent 2"],fontsize = 12)
    plt.show()