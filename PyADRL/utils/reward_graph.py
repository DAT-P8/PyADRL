import matplotlib.pyplot as plt
from pathlib import Path
from numpy import mean


def create_reward_graph(
    rewards,
    plot_best: bool = False,
    figure_path: Path | None = None,
):
    """Plots agent rewards.
    Args:
        rewards: The dictionary mapping agents to a list of rewards.
        plot_best: If set to true only the pursuer and evader with the highest mean reward will be plottet.
        figure_path: If set the plot will be saved to the destination instead of being plottet
    """
    # Todo: fix iterations and agent rewards
    reward_len = 0
    for v in rewards.values():
        reward_len = len(v)
    iterations = list(range(1, reward_len + 1))

    _fig, ax = plt.subplots(figsize=(10, 5))

    if plot_best:
        best_pursuer = find_best("pursuer", rewards)
        best_evader = find_best("evader", rewards)
        ax.plot(iterations, rewards[best_pursuer], label=best_pursuer, linewidth=2)
        ax.plot(iterations, rewards[best_evader], label=best_evader, linewidth=2)
    else:
        for agent, agent_reward in rewards.items():
            ax.plot(iterations, agent_reward, label=agent, linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.set_title("Mean reward per Iteration")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if figure_path is not None:
        plt.savefig(f"{figure_path}/reward_graph.svg")
    else:
        plt.show()


def find_best(policy: str, rewards: dict) -> str:
    return max(
        (agent for agent in rewards if policy in agent),
        key=lambda agent: mean(rewards[agent]),
    )
