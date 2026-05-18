import ray
import time
import json
from pathlib import Path
from collections import defaultdict
from ray.tune.registry import _global_registry, ENV_CREATOR
from ..envs.reward_functions.grid_world_rewards import GridWorldRewards
from .trainables.alternate_training import _run_alternating_loop
from .trainables.iterative_training import _run_iterative_loop
from ..utils.register_env import _register_gridworld_env
from ..utils.config_builder import _build_ppo_config
from ..utils.reward_graph import create_reward_graph
from ..utils.map_load import load_map_dict

# Callbacks
from ..logger.metrics import MetricsCallback
from ..logger.heatmaps import HeatmapCallback

# Trainables
ALTERNATING = "alternate"
SIMULTANEOUS = "simultaneous"

DEFAULT_TRAINING = {
    "name": ALTERNATING,
    "n_stages": 4,
    "iters_per_stage": 20,
}

DEFAULT_CONFIG = {
    # --- Training params ---
    "lr": 3e-4,
    "gamma": 0.99,
    "lambda_": 0.95,
    "clip_param": 0.2,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    # --- Architecture params ---
    "train_batch_size": 10000,
    "minibatch_size": 512,
    "num_epochs": 10,
    # --- Resource params ---
    # Defaults for standalone gridworld_train calls. Note: when called from
    # gridworld_tune, model_config is the trial.config so these are overridden.
    "num_learners": 1,
    "num_env_runners": 4,
    "num_envs_per_env_runner": 5,
}


def gridworld_train(
    map: str,
    n_pursuers: int = 2,
    n_evaders: int = 1,
    training_config: dict = DEFAULT_TRAINING,
    model_config: dict = DEFAULT_CONFIG,
    training_path: Path | None = None,
    shielding: bool = False,
):
    ray.shutdown()
    ray.init()

    map_dict = load_map_dict(map)

    # Only register the environment if it hasn't been registered
    if not _global_registry.contains(ENV_CREATOR, "gridworld"):
        _register_gridworld_env(
            map_dict=map_dict,
            reward_function=GridWorldRewards(),
            n_pursuers=n_pursuers,
            n_evaders=n_evaders,
            shielding=shielding,
        )

    figure_path = None
    model_path = None
    if training_path:
        figure_path = training_path / "figures"
        model_path = training_path / "models"

    timer = Timer()
    ppo_config = _build_ppo_config(
        config=model_config,
        callbacks=[MetricsCallback],
        env_config=map_dict,
        n_pursuers=n_pursuers,
        n_evaders=n_evaders,
        figure_path=figure_path,
        metrics_path=training_path,
    )
    algo = ppo_config.build_algo()

    if training_config["name"] == SIMULTANEOUS:
        timer.start()
        _run_iterative_loop(
            algo=algo,
            iterations=training_config["n_iterations"],
            model_path=model_path,
        )
        timer.stop()
    elif training_config["name"] == ALTERNATING:
        timer.start()
        _run_alternating_loop(
            algo=algo,
            n_stages=training_config["n_stages"],
            iters_per_stage=training_config["iters_per_stage"],
            model_path=model_path,
        )
        timer.stop()
    else:
        raise ValueError(f"Recieved unkown trainable name {training_config['type']}")

    if training_path:
        timer.write_time(training_path)
    else:
        timer.print_time()

    # Handle all metric and saving stuff here
    rewards = get_rewards(training_path)
    create_reward_graph(rewards=rewards, figure_path=figure_path)

    # Save final model to easily find it and evaluate it
    if training_path:
        final_model = training_path / "final_model"
        print(f"Saving final model at {final_model}")
        algo.save(str(final_model))
        evaluate_model(
            training_path=training_path,
            model_config=model_config,
            map_dict=map_dict,
            figure_path=figure_path,
            n_pursuers=n_pursuers,
            n_evaders=n_evaders,
        )


def evaluate_model(
    training_path, model_config, map_dict, figure_path, n_pursuers, n_evaders
):
    evaluation_duration = 1000
    model_config = dict(model_config)
    model_config["evaluation_num_env_runners"] = 1
    model_config["evaluation_duration"] = evaluation_duration
    callbacks = [MetricsCallback, HeatmapCallback]
    ppo_config = _build_ppo_config(
        config=model_config,
        callbacks=callbacks,
        env_config=map_dict,
        n_pursuers=n_pursuers,
        n_evaders=n_evaders,
        figure_path=figure_path,
        metrics_path=training_path,
    )

    algo = ppo_config.build_algo()
    algo.restore(str(training_path / "final_model"))

    print(f"Evaluating model over {evaluation_duration} episodes")
    algo.evaluate()


class Timer:
    def start(self) -> None:
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.training_time = self.end_time - self.start_time

    def write_time(self, path: Path):
        time_info = {
            "Training start": self.start_time,
            "Training end": self.end_time,
            "Training time": self.training_time,
        }
        time_file = path / "training_time.json"
        with open(time_file, "w") as f:
            json.dump(time_info, f, indent=2)

    def print_time(self):
        print(f"Training time: {self.training_time}")


def get_rewards(training_path):
    # Read the JSON file
    file_path = training_path / "evaluation_metrics.json"
    with open(str(file_path), "r") as f:
        data = json.load(f)

    # Create a dictionary to store rewards by agent
    rewards_by_agent = defaultdict(list)

    # Iterate through each entry and collect rewards
    for entry in data:
        mean_rewards = entry.get("mean_rewards", {})
        for agent_name, reward in mean_rewards.items():
            rewards_by_agent[agent_name].append(reward)

    # Convert to regular dict (optional)
    rewards_by_agent = dict(rewards_by_agent)
    return rewards_by_agent
