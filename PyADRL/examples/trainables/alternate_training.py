import random
import os
from ray import tune
from ray.rllib.callbacks.callbacks import RLlibCallback
from ...utils.config_builder import _build_ppo_config


EVADER = "evader"
PURSUER = "pursuer"


def _run_alternating_loop(
    algo,  # type: ignore[no-untyped-def]
    n_stages: int,
    iters_per_stage: int,
    report_to_tune: bool = False,
    checkpoint_dir: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Core alternating self-play training loop.

    Used by both gridworld_train (with checkpointing/metrics) and
    gridworld_tune (with tune.report). Keeps the loop logic in one place.

    Args:
        algo: Built RLlib Algorithm instance.
        n_stages: Number of alternating stages.
        iters_per_stage: PPO iterations per phase within each stage.
        report_to_tune: If True, call tune.report after each iteration.
        checkpoint_dir: If set, save checkpoints after each full stage.

    Returns:
        Tuple of (rewards list, episodes_data list).
    """
    rewards: list[dict] = []
    episodes_data: list[dict] = []
    global_step = 0

    pools = {EVADER: [], PURSUER: []}
    train_evader = True
    for k in range(n_stages * 2):
        training, frozen = alternate(train_evader)
        print(f"Stage {k + 1}: training {training}")

        assert algo.learner_group is not None
        assert algo.config is not None

        # Unfreeze, update algo-level policies_to_train, refreeze
        algo.config._is_frozen = False
        algo.config.multi_agent(policies_to_train=[f"{training}_policy"])
        algo.config._is_frozen = True

        # Also update each learner's config so gradient updates are gated correctly
        algo.learner_group.foreach_learner(
            lambda learner, *_args: learner.config.multi_agent(
                policies_to_train=[f"{training}_policy"]
            )
        )

        # If pool has past policies, sample and load into frozen policy.
        if any(pools[frozen]):
            opp_weights = sample_opponent(pools[frozen])
            algo.learner_group.set_weights({f"{frozen}_policy": opp_weights})
            # Sync weights to env runners so rollouts use the correct opponent.
            if algo.env_runner_group is not None and algo.config.num_env_runners > 0:
                algo.env_runner_group.sync_weights(
                    from_worker_or_learner_group=algo.learner_group,
                    policies=[f"{frozen}_policy"],
                )

        for i in range(iters_per_stage):
            result = algo.train()
            mean = result["env_runners"]["agent_episode_returns_mean"]
            rewards.append(mean)
            global_step += 1

        assert algo.learner_group is not None
        updated_weights = algo.learner_group.get_weights()[f"{training}_policy"]
        pools[training].append(updated_weights)

        # Change weights of frozen policy back to the most trained ones
        if len(pools[frozen]) != 0:
            algo.learner_group.set_weights({f"{frozen}_policy": pools[frozen][-1]})
        print(f"Evaluating stage {k + 1}: {training}")
        eval_result = algo.evaluate()

        # Report to Tune so ASHA can prune bad trials early
        if report_to_tune:
            eval_mean = eval_result["env_runners"]["agent_episode_returns_mean"]
            total_mean_reward = (
                sum(eval_mean.values()) if isinstance(eval_mean, dict) else eval_mean
            )
            tune.report(metrics={"mean_reward": total_mean_reward})

        # Save a checkpoint after each full stage (evader+pursuer training)
        if checkpoint_dir and training == PURSUER:
            print(f"Saving stage {k + 1} at {checkpoint_dir}/cp_{k + 1:05d}")
            check = os.path.abspath(f"{checkpoint_dir}/cp_{k + 1:05d}")
            algo.save(checkpoint_dir=check)

        # Alternate what policy gets trained
        train_evader = not train_evader

    return rewards, episodes_data


# ---------------------------------------------------------------------------
# Ray Tune hyperparameter search
# ---------------------------------------------------------------------------
def alternate_trainable(
    config: dict,
    checkpoint_dir: str | None = None,
    n_stages: int = 4,
    iters_per_stage: int = 20,
    callbacks: list[RLlibCallback] = [],
) -> None:  # type: ignore[type-arg]
    """Trainable function for Ray Tune.

    Each trial builds an algo with sampled hyperparameters, runs the full
    alternating self-play loop, and reports metrics back to Tune after every
    training iteration so ASHA can prune underperforming trials early.
    """
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
        callbacks=callbacks,
    )

    # Configuration for evaluating policies after a stage
    ppo_config = ppo_config.evaluation(
        evaluation_interval=None,
        evaluation_num_env_runners=0,
        evaluation_duration=20,
    )

    algo = ppo_config.build_algo()
    try:
        _run_alternating_loop(
            algo,
            n_stages=n_stages,
            iters_per_stage=iters_per_stage,
            report_to_tune=True,
            checkpoint_dir=checkpoint_dir,
        )
    finally:
        algo.stop()


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def sample_opponent(pool: list[dict], p_old: float = 0.3) -> dict:
    """With prob P_OLD sample a random old policy, otherwise use the latest."""
    if len(pool) == 1:
        return pool[-1]  # most recent
    elif random.random() < p_old:
        return random.choice(pool[:-1])  # Sample from all but the last policy
    else:
        return pool[-1]


def alternate(train_evader: bool) -> tuple[str, str]:
    return (EVADER, PURSUER) if train_evader else (PURSUER, EVADER)
