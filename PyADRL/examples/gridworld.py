import grpc
import ray
import os
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ..envs.gridworld_env import GridWorldEnvironment
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from pprint import pprint
import matplotlib.pyplot as plt
from ..utils.custom_logger import CustomLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np

from ray import tune
from ray.tune import Tuner, CLIReporter
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
from ray.rllib.algorithms.algorithm import Algorithm

def gridworld_train(checkpoint_path: str | None = None):
    ray.init(log_to_driver=False)

    register_env(
        "gridworld",
        lambda cfg: ParallelPettingZooEnv(
            GridWorldEnvironment(channel=grpc.insecure_channel("localhost:50051"))
        ),
    )

    config: PPOConfig = (
        PPOConfig()
        .environment("gridworld")
        .multi_agent(
            policies={"pursuer_policy", "evader_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                "pursuer_policy" if "pursuer" in str(agent_id) else "evader_policy"
            ),
        )
        .learners(
            num_learners=5,
        )
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=5,
        )
        .training(
            train_batch_size=10000,  # Larger batches = more stable gradients
            minibatch_size=512,
            num_epochs=10,  # More SGD passes per batch
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,  # Encourage exploration
        )
        .evaluation(evaluation_num_env_runners=0)
    )

    algo = config.build_algo()

    if checkpoint_path is not None:
        print("Restoring checkpoint from checkpoint:", checkpoint_path)
        algo.restore(checkpoint_path)

    rewards = []

    for i in range(10):
        result = algo.train()

        checkpoint_dir = os.path.abspath(f"./checkpoints/iter_{i + 1}")
        algo.save(checkpoint_dir=checkpoint_dir)

        mean = result["env_runners"]["agent_episode_returns_mean"]
        rewards.append(mean)

    iterations = list(range(1, len(rewards) + 1))
    evader_rewards = [r["evader"] for r in rewards]
    pursuer_0_rewards = [r["pursuer_0"] for r in rewards]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(iterations, evader_rewards, label="Evader", color="royalblue", linewidth=2)
    ax.plot(
        iterations, pursuer_0_rewards, label="Pursuers", color="seagreen", linewidth=2
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.set_title("Mean reward per Iteration")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


# for some reason this function creates 2 enviorments?
def gridworld_test(checkpoint_path: str):
    # Load the algorithm from checkpoint
    ray.init(num_gpus = 0)

    register_env(
        "gridworld",
        lambda cfg: ParallelPettingZooEnv(
            GridWorldEnvironment(
                channel=grpc.insecure_channel("localhost:50051"), step_delay=0.2
            )
        ),
    )

    config: PPOConfig = (
        PPOConfig()
        .callbacks(CustomLoggerCallback)
        .environment("gridworld")
        .multi_agent(
            policies={"pursuer_policy", "evader_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                "pursuer_policy" if "pursuer" in str(agent_id) else "evader_policy"
            ),
        )
        .env_runners(
            num_env_runners=1,
        )
        .evaluation(evaluation_num_env_runners=1, evaluation_interval=1)
    )

    algo = config.build_algo()
    algo.restore(os.path.abspath("./checkpoints/iter_100"))
    results = algo.evaluate()
    pprint(results)
    algo.stop()







class RewardCallback(DefaultCallbacks):
    """
    Custom callback class for recording mean reward of each policy after each episode
    """
    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        rewards_dict = result['env_runners']['agent_episode_returns_mean']
        for agent_id, mean_reward in rewards_dict.items():
            metric_name = f"mean_reward/{agent_id}"

            #Using metrics_logger in on_train_result acts weird and only logs the newest value even if reduce="item_series"
            #This code ensures rewards are logged as an array so train history is available for plotting and evaluation
            if metric_name in result.keys():
                rewards = result[metric_name]
                rewards = np.append(rewards, mean_reward)
            else:
                rewards = np.array(mean_reward)

            metrics_logger.log_value(
                key=metric_name,
                value=rewards,
                reduce="item_series",
            )


def gridworld_tuner_train():
    register_gridworld_env()

    config: PPOConfig = (
        PPOConfig()
        .environment("gridworld")
        .multi_agent(
            policies={"pursuer_policy", "evader_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                "pursuer_policy" if "pursuer" in str(agent_id) else "evader_policy"
            ),
        )
        .learners(
            num_learners=5,
        )
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=5,
        )
        .training(
            train_batch_size=10000,  # Larger batches = more stable gradients
            minibatch_size=512,
            num_epochs=10,  # More SGD passes per batch
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,  # Encourage exploration
        )
        .env_runners(enable_connectors=False)
        .callbacks(RewardCallback)
        .reporting(keep_per_episode_custom_metrics=True)
    )

    stopping_criteria = {"training_iteration": 10}

    tuner = Tuner(
        "PPO",
        param_space=config,
        run_config=tune.RunConfig(
            name="gridworld",
            storage_path=os.path.abspath("./checkpoints"),
            stop=stopping_criteria,
        ),
    )
    result = tuner.fit().get_best_result()
    metrics = result.metrics
    mean_rewards = {key: value for key,
                     value in metrics.items() if ("mean_reward/" in key)}
    
    fig, ax = plt.subplots(figsize=(10, 5))

    for key, rewards in mean_rewards.items():
        agent = key.replace("mean_reward/", "")
        length = list(range(1, len(rewards)+1))
        ax.plot(length, rewards, label=agent, linewidth=2)
        
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.set_title("Mean reward per Iteration")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

def gridworld_tuner_test(checkpoint, num_episodes = 10):
    register_gridworld_env()
    model = Algorithm.from_checkpoint(checkpoint)
    #Change config of loaded algorithm to only spawn 1 set of agents
    
    env = ParallelPettingZooEnv(
        GridWorldEnvironment(channel=grpc.insecure_channel("localhost:50051"))
    )

    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = model.compute_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        print(f"Episode_{i} had reward {reward}")


        #done = {"__all__": False}
        #while not done["__all__"]:
        #    actions = {}
        #    for agent_id, agent_obs in obs.items():
        #        actions[agent_id] = model.compute_single_action(agent_obs, policy_id=agent_id)
        #    obs, rewards, done, info = env.step(actions)
    

def register_gridworld_env():
    register_env(
        "gridworld",
        lambda cfg: ParallelPettingZooEnv(
            GridWorldEnvironment(channel=grpc.insecure_channel("localhost:50051"))
        ),
    )
