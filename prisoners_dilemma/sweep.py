#%%
import numpy as np
from utils import *
import os

os.mkdir("stochastic")


def run_dual_sweep_stochastic(
    T,
    num_trials,
    alphas=[1, 3],
    lrs1=np.linspace(0.01, 0.6, 100)[25:],
    lrs2=np.linspace(0.01, 0.6, 100),
):
    actions_over_time_all = np.zeros((T, 2, len(lrs1), len(lrs2), len(alphas)))
    # B1_over_time_all = np.zeros(
    #     (T, 4, 4, 2, 2, len(lrs1), len(lrs2), len(alphas), num_trials)
    # )
    # q_pi_over_time_all = np.zeros(
    #     (T, 2, 2, len(lrs1), len(lrs2), len(alphas), num_trials)
    # )

    for a, alpha in enumerate(alphas):
        for k, lr_pB_1 in enumerate(lrs1):
            if not os.path.exists(f"stochastic/{lr_pB_1}"):
                os.mkdir(f"stochastic/{lr_pB_1}")
            print(f"lr = : {lr_pB_1}")
            for j, lr_pB_2 in enumerate(lrs2):
                if os.path.exists(f"stochastic/{lr_pB_1}/{lr_pB_2}"):
                    continue
                os.mkdir(f"stochastic/{lr_pB_1}/{lr_pB_2}")

                print(f"lr2 = : {lr_pB_2}")

                collect = []

                for t in range(num_trials):

                    alpha_1 = np.random.normal(alpha, 0.15)
                    alpha_2 = np.random.normal(alpha, 0.15)
                    agent_1, agent_2, D = construct(
                        lr_pB=lr_pB_1, lr_pB_2=lr_pB_2, factors_to_learn="all"
                    )
                    agent_1.action_selection = "stochastic"
                    agent_2.action_selection = "stochastic"
                    agent_1.alpha = alpha_1
                    agent_2.alpha = alpha_2
                    actions_over_time = run_sim_collect_actions(
                        agent_1, agent_2, observation_1=[0], observation_2=[0], D=D, T=T
                    )
                    collect.append(actions_over_time)

                actions_over_time_all[:, :, k, j, a] = np.mean(
                    np.array(collect), axis=0
                )
                # B1_over_time_all[:, :, :, :, :, k, j, a, t] = B1_over_time
                # q_pi_over_time_all[:, :, :, k, j, a, t] = q_pi_over_time
                np.save(
                    f"stochastic/{lr_pB_1}/{lr_pB_2}/actions_over_time_all",
                    actions_over_time_all,
                    allow_pickle=True,
                )
    # np.save("stochastic/B1_over_time_all", B1_over_time_all, allow_pickle=True)
    # np.save("stochastic/q_pi_over_time_all", q_pi_over_time_all, allow_pickle=True)


def run_dual_sweep_deterministic(
    T,
    lrs1=np.linspace(0.01, 0.6, 100),
    lrs2=np.linspace(0.01, 0.6, 100),
    only_actions=False,
):
    actions_over_time_all = np.zeros((T, 2, len(lrs1), len(lrs2)))
    B1_over_time_all = np.zeros((T, 4, 4, 2, 2, len(lrs1), len(lrs2)))
    q_pi_over_time_all = np.zeros((T, 2, 2, len(lrs1), len(lrs2)))

    for k, lr_pB_1 in enumerate(lrs1):
        print(f"lr = : {lr_pB_1}")
        for j, lr_pB_2 in enumerate(np.linspace(0.01, 0.6, 100)):

            agent_1, agent_2, D = construct(
                lr_pB=lr_pB_1, lr_pB_2=lr_pB_2, factors_to_learn="all"
            )
            if only_actions:

                actions_over_time = run_sim_collect_actions(
                    agent_1, agent_2, observation_1=[0], observation_2=[0], D=D, T=T
                )
                actions_over_time_all[:, :, k, j] = actions_over_time
            else:
                (
                    actions_over_time,
                    B1_over_time,
                    q_pi_over_time,
                    q_s_over_time,
                    agent_1,
                ) = run_sim_collect_all_data(
                    agent_1, agent_2, observation_1=[0], observation_2=[0], D=D, T=T
                )
                actions_over_time_all[:, :, k, j] = actions_over_time
                B1_over_time_all[:, :, :, :, :, k, j] = B1_over_time
                q_pi_over_time_all[:, :, :, k, j] = q_pi_over_time

    np.save("actions_over_time_all", actions_over_time_all, allow_pickle=True)
    if not only_actions:
        np.save("B1_over_time_all", B1_over_time_all, allow_pickle=True)
        np.save("q_pi_over_time_all", q_pi_over_time_all, allow_pickle=True)


run_dual_sweep_stochastic(1000, 100)
