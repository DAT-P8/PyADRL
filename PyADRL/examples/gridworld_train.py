import random
import grpc
import ray
import os
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ..envs.ngw_env import NGWEnvironment
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ..envs.map_configs.square_map import SquareMapConfig
from ..envs.reward_functions.grid_world_rewards import GridWorldRewards
from ray.tune.registry import register_env
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ..logger.metricslogger import (
    MetricsCallback,
    build_train_iteration_data,
    build_train,
    metrics_path,
    write_metrics,
)

import matplotlib.pyplot as plt
from ..utils.model_save import restore_training, setup_checkpoint_dir

# Probability of sampling an old opponent policy
P_OLD = 0.3

# Number of alternating stages and PPO iterations per stage
N_STAGES = 4
ITERS_PER_STAGE = 20

# Total training iterations across all stages (2 phases per stage: evader + pursuer)
TOTAL_TUNE_ITERATIONS = N_STAGES * ITERS_PER_STAGE * 2


def _register_gridworld_env() -> None:
    """Register the gridworld environment with Ray. Safe to call multiple times."""
    register_env(
        "gridworld",
        lambda _cfg: ParallelPettingZooEnv(
            NGWEnvironment(
                channel=grpc.insecure_channel("localhost:50051"),
                map_config=SquareMapConfig(11, 11, 5, 5),
                reward_function=GridWorldRewards(),
                n_pursuers=2,
                n_evaders=1,
            )
        ),
    )


def _build_ppo_config(
    lr: float = 3e-4,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    clip_param: float = 0.2,
    vf_loss_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    train_batch_size: int = 10000,
    minibatch_size: int = 512,
    num_epochs: int = 10,
    num_learners: int = 2,
    num_env_runners: int = 4,
    num_envs_per_env_runner: int = 5,
) -> PPOConfig:
    """Build a PPOConfig with the given hyperparameters.

    Shared between gridworld_train and gridworld_tune to keep config in sync.
    """
    return (
        PPOConfig()
        .environment("gridworld")
        .multi_agent(
            policies={"pursuer_policy": PolicySpec(), "evader_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *_args, **_kwargs: (
                "pursuer_policy" if "pursuer" in str(agent_id) else "evader_policy"
            ),
        )
        .learners(
            num_learners=num_learners,
        )
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
        )
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            lr=lr,
            gamma=gamma,
            lambda_=lambda_,
            clip_param=clip_param,
            vf_loss_coeff=vf_loss_coeff,
            entropy_coeff=entropy_coeff,
        )
        .callbacks(MetricsCallback)
        .evaluation(evaluation_num_env_runners=0)
    )


def sample_opponent(pool: list[dict]) -> dict:
    """With prob P_OLD sample a random old policy, otherwise use the latest."""
    if len(pool) == 1:
        return pool[-1]  # most recent
    elif random.random() < P_OLD:
        return random.choice(pool[:-1])  # Sample from all but the last policy
    else:
        return pool[-1]


def _run_alternating_loop(
    algo,  # type: ignore[no-untyped-def]
    n_stages: int,
    iters_per_stage: int,
    pursuer_pool: list[dict],
    evader_pool: list[dict],
    report_to_tune: bool = False,
    checkpoint_dir: str | None = None,
    start_stage: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Core alternating self-play training loop.

    Used by both gridworld_train (with checkpointing/metrics) and
    gridworld_tune (with tune.report). Keeps the loop logic in one place.

    Args:
        algo: Built RLlib Algorithm instance.
        n_stages: Number of alternating stages.
        iters_per_stage: PPO iterations per phase within each stage.
        pursuer_pool: Mutable list of pursuer weight snapshots.
        evader_pool: Mutable list of evader weight snapshots.
        report_to_tune: If True, call tune.report after each iteration.
        checkpoint_dir: If set, save checkpoints after each full stage.
        start_stage: Stage index to start from (for checkpoint resumption).

    Returns:
        Tuple of (rewards list, episodes_data list).
    """
    rewards: list[dict] = []
    episodes_data: list[dict] = []
    train_metrics_path = metrics_path("train") if not report_to_tune else None
    global_step = 0

    for k in range(start_stage, start_stage + n_stages):
        for training_policy, frozen_policy, label, pool, opp_pool in [
            (
                "evader_policy",
                "pursuer_policy",
                "evader",
                evader_pool,
                pursuer_pool,
            ),
            (
                "pursuer_policy",
                "evader_policy",
                "pursuer",
                pursuer_pool,
                evader_pool,
            ),
        ]:
            print(f"\nStage {k + 1}: training {label}")

            assert algo.learner_group is not None
            assert algo.config is not None

            # Unfreeze, update algo-level policies_to_train, refreeze
            algo.config._is_frozen = False
            algo.config.multi_agent(policies_to_train=[training_policy])
            algo.config._is_frozen = True

            # Also update each learner's config so gradient updates are gated correctly
            algo.learner_group.foreach_learner(
                lambda learner, *_args: learner.config.multi_agent(
                    policies_to_train=[training_policy]
                )
            )

            # If pool has past policies, sample and load into frozen policy.
            if opp_pool:
                opp_weights = sample_opponent(opp_pool)
                algo.learner_group.set_weights({frozen_policy: opp_weights})
                # Sync weights to env runners so rollouts use the correct opponent.
                # When num_env_runners=0 (all in-process), the local runner picks up
                # weights automatically via the learner group — no sync needed.
                if (
                    algo.env_runner_group is not None
                    and algo.config.num_env_runners > 0
                ):
                    algo.env_runner_group.sync_weights(
                        from_worker_or_learner_group=algo.learner_group,
                        policies=[frozen_policy],
                    )

            for i in range(iters_per_stage):
                result = algo.train()
                mean = result["env_runners"]["agent_episode_returns_mean"]
                rewards.append(mean)
                global_step += 1

                # Metrics logging for standard training
                if not report_to_tune and train_metrics_path is not None:
                    iteration_data = build_train_iteration_data(result, i + 1)
                    episodes_data.extend(iteration_data.get("episodes", []))
                    write_metrics(train_metrics_path, {"episodes": episodes_data})

                # Report to Tune so ASHA can prune bad trials early
                if report_to_tune:
                    total_mean_reward = (
                        sum(mean.values()) if isinstance(mean, dict) else mean
                    )
                    tune.report(
                        metrics={
                            "mean_reward": total_mean_reward,
                            "agent_rewards": mean,
                            "stage": k + 1,
                            "phase": label,
                            "training_iteration": global_step,
                        },
                    )

            assert algo.learner_group is not None
            updated_weights = algo.learner_group.get_weights()[training_policy]
            pool.append(updated_weights)

            # Save a checkpoint after each full stage (evader+pursuer training)
            if checkpoint_dir and label == "pursuer":
                print(f"Saving stage {k + 1} at {checkpoint_dir}/cp_{k + 1:05d}")
                check = os.path.abspath(f"{checkpoint_dir}/cp_{k + 1:05d}")
                tune.report(checkpoint=check)
                # algo.save(checkpoint_dir=check)

    return rewards, episodes_data


# ---------------------------------------------------------------------------
# Ray Tune hyperparameter search
# ---------------------------------------------------------------------------


def _tune_trainable(config: dict) -> None:  # type: ignore[type-arg]
    """Trainable function for Ray Tune.

    Each trial builds an algo with sampled hyperparameters, runs the full
    alternating self-play loop, and reports metrics back to Tune after every
    training iteration so ASHA can prune underperforming trials early.
    """
    _register_gridworld_env()

    ppo_config = _build_ppo_config(
        lr=config["lr"],
        gamma=config["gamma"],
        lambda_=config["lambda_"],
        clip_param=config["clip_param"],
        vf_loss_coeff=config["vf_loss_coeff"],
        entropy_coeff=config["entropy_coeff"],
        train_batch_size=config["train_batch_size"],
        minibatch_size=config["minibatch_size"],
        num_epochs=config["num_epochs"],
        # All in-process: no remote actors, no placement group conflicts
        num_learners=config.get("num_learners", 0),
        num_env_runners=config.get("num_env_runners", 0),
        num_envs_per_env_runner=config.get("num_envs_per_env_runner", 10),
    )

    algo = ppo_config.build_algo()
    try:
        _run_alternating_loop(
            algo,
            n_stages=N_STAGES,
            iters_per_stage=ITERS_PER_STAGE,
            pursuer_pool=[],
            evader_pool=[],
            report_to_tune=True,
        )
    finally:
        algo.stop()


def gridworld_tune(
    num_samples: int = 25,
    max_concurrent_trials: int = 4,
    width: int = 11,
    height: int = 11,
    target_x: int = 5,
    target_y: int = 5,
    checkpoint: str | None = None,
) -> tune.ResultGrid:
    """Run a Ray Tune hyperparameter search over the alternating self-play loop.

    Args:
        num_samples: Number of hyperparameter configurations to try.
        max_concurrent_trials: Max trials running in parallel. Lower this if
            you're memory-constrained (each trial spins up learners + env runners).

    Returns:
        tune.ResultGrid with all trial results.
    """
    ray.shutdown()
    ray.init(log_to_driver=False)

    _register_gridworld_env()

    search_space = {
        # --- Training params ---
        "lr": tune.loguniform(1e-5, 1e-3),
        "gamma": tune.uniform(0.95, 0.999),
        "lambda_": tune.uniform(0.9, 1.0),
        "clip_param": tune.uniform(0.1, 0.3),
        "vf_loss_coeff": tune.uniform(0.25, 1.0),
        "entropy_coeff": tune.loguniform(0.001, 0.05),
        # --- Architecture params ---
        "train_batch_size": tune.choice([5000, 10000, 20000]),
        "minibatch_size": tune.choice([256, 512, 1024]),
        "num_epochs": tune.choice([5, 10, 15, 20]),
        # --- Resource params (all in-process to avoid placement group errors) ---
        # num_learners=0 and num_env_runners=0 means the trial driver handles
        # both training and environment stepping — no remote actors needed,
        # so no placement group bundle conflicts.
        "num_learners": 0,
        "num_env_runners": 0,
        "num_envs_per_env_runner": 10,
    }

    scheduler = ASHAScheduler(
        metric="mean_reward",
        mode="max",
        max_t=TOTAL_TUNE_ITERATIONS,
        grace_period=ITERS_PER_STAGE,  # Let each trial finish at least one full phase
        reduction_factor=3,  # Keep top 1/3 of trials at each rung
    )

    tuner = tune.Tuner(
        _tune_trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=tune.RunConfig(
            name="gridworld_tune",
            storage_path="./checkpoints",
            verbose=2,
        ),
    )

    results = tuner.fit()

    try:
        best = results.get_best_result(metric="mean_reward", mode="max")
        if best is not None:
            print("\n=== Best trial ===")
            print(f"Config:  {best.config}")
            best_reward = (
                best.metrics.get("mean_reward", "N/A") if best.metrics else "N/A"
            )
            print(f"Reward:  {best_reward}")
            print(f"Log dir: {best.path}")
    except RuntimeError:
        print("\nNo successful trials. Check error logs above for details.")

    ray.shutdown()
    return results


# ---------------------------------------------------------------------------
# Standard training (fixed hyperparameters, with checkpointing + plotting)
# ---------------------------------------------------------------------------


def gridworld_train(
    checkpoint: str | None = None,
    model_name: str | None = None,
    width: int = 11,
    height: int = 11,
    target_x: int = 5,
    target_y: int = 5,
):
    # If Ray is already initialized from a previous run, shut it down before starting a new one.
    ray.shutdown()
    ray.init()

    register_env(
        "gridworld",
        lambda cfg: ParallelPettingZooEnv(
            NGWEnvironment(
                channel=grpc.insecure_channel("localhost:50051"),
                map_config=SquareMapConfig(width, height, target_x, target_y),
                reward_function=GridWorldRewards(),
                n_pursuers=2,
                n_evaders=1,
            )
        ),
    )

    config: PPOConfig = (
        PPOConfig()
        .environment(
            "gridworld",
            env_config={
                "map_width": width,
                "map_height": height,
                "target_x": target_x,
                "target_y": target_y,
            },
        )
        .multi_agent(
            policies={"pursuer_policy", "evader_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                "pursuer_policy" if "pursuer" in str(agent_id) else "evader_policy"
            ),
        )
        .learners(
            num_learners=2,  # Number of parallel learner processes for computing gradients
        )
        .env_runners(
            num_env_runners=4,  # Number of processes/threads that run the environment in parallel
            num_envs_per_env_runner=5,  # Number of environments per env_runner
        )
        .training(
            train_batch_size=10000,  # Number of timesteps before each gradient update. Larger batches = more stable gradients
            minibatch_size=512,  # Size of each mini batch for each SGD update
            num_epochs=10,  # Number of full passes over the train batch per learner. More epochs = more gradient updates per batch
            lr=3e-4,  # Learning rate for optimization
            gamma=0.99,  # Discount factor: future rewards are multiplied by gamma
            lambda_=0.95,  # Balances short-term, low-variance estimates against long-term, high-variance returns in GAE (General Advantage Estimation)
            clip_param=0.2,  # The PPO clip parameter: Limits how much the policy can change in one update
            vf_loss_coeff=0.5,  # Weight of the value function loss in the total loss
            entropy_coeff=0.01,  # Encourage exploration
        )
        .callbacks(MetricsCallback)
        .evaluation(
            evaluation_num_env_runners=0
        )  # No separate evaluation environments. >0 = parallel evaluation of policy while training
    )
    _register_gridworld_env()

    config = _build_ppo_config()
    algo = config.build_algo()

    pursuer_pool: list[dict] = []
    evader_pool: list[dict] = []
    start_iteration = 0
    checkpoint_dir = ""

    train_metrics_path = metrics_path("train")
    if checkpoint is not None:
        checkpoint_dir, start_iteration = restore_training(
            algo, checkpoint, pursuer_pool, evader_pool, model_name=model_name
        )
    else:
        checkpoint_dir = setup_checkpoint_dir(model_name=model_name)
        print(f"No checkpoint specified. Starting new training run at {checkpoint_dir}")

    rewards: list[dict] = []
    episodes_data: list[dict] = []

    # try/finally ensures Ray always shuts down cleanly even if training crashes
    try:
        rewards, episodes_data = _run_alternating_loop(
            algo,
            n_stages=N_STAGES,
            iters_per_stage=ITERS_PER_STAGE,
            pursuer_pool=pursuer_pool,
            evader_pool=evader_pool,
            report_to_tune=False,
            checkpoint_dir=checkpoint_dir,
            start_stage=start_iteration,
        )
    finally:
        algo.stop()
        ray.shutdown()

        # Add final aggregate summary after all episodes are complete.
        write_metrics(
            train_metrics_path,
            build_train(episodes_data, final_rewards=rewards[-1] if rewards else {}),
        )

        iterations = list(range(1, len(rewards) + 1))
        evader_rewards = [r["evader_2"] for r in rewards]
        pursuer_0_rewards = [r["pursuer_0"] for r in rewards]

        _fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            iterations, evader_rewards, label="Evader", color="royalblue", linewidth=2
        )
        ax.plot(
            iterations,
            pursuer_0_rewards,
            label="Pursuers",
            color="seagreen",
            linewidth=2,
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        ax.set_title("Mean reward per Iteration")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()
