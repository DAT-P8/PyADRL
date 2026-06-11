import random
import numpy as np
import torch
from pathlib import Path
from ray import tune
from ray.rllib.callbacks.callbacks import RLlibCallback
from ...utils.config_builder import _build_ppo_config
from ...logger.metrics import summarize_evaluation, extract_entropies


EVADER = "evader"
PURSUER = "pursuer"


def _run_alternating_loop(
    algo,  # type: ignore[no-untyped-def]
    n_stages: int,
    iters_per_stage: int,
    report_to_tune: bool = False,
    model_path: Path | None = None,
) -> dict:
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
        Result dictionary from the last training
    """
    global_step = 0
    result = {}

    # Last seen mean policy entropy per module. Updated every algo.train()
    # call; only the policy being trained that stage produces a fresh value,
    # so the frozen side carries its value from its own last training phase.
    last_entropy: dict[str, float] = {}

    # Pull n_evaders + time_limit once — needed by summarize_evaluation for
    # capture/ACS normalisation. Read here so the eval-summary call site
    # doesn't have to know about RLlib config plumbing.
    n_evaders = 1
    time_limit = 100
    if algo.config is not None and algo.config.env_config is not None:
        n_evaders = algo.config.env_config.get("n_evaders", 1)
        time_limit = algo.config.env_config.get("time_limit", 100)

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
            _set_policy_weights(algo, f"{frozen}_policy", opp_weights)

        # Train stage
        for i in range(iters_per_stage):
            result = algo.train()
            global_step += 1
            last_entropy.update(extract_entropies(result))

        assert algo.learner_group is not None
        updated_weights = algo.learner_group.get_weights()[f"{training}_policy"]
        pools[training].append(updated_weights)

        # Change weights of frozen policy back to the most trained ones
        if len(pools[frozen]) != 0:
            _set_policy_weights(algo, f"{frozen}_policy", pools[frozen][-1])
        print(f"Evaluating stage {k + 1}: {training}")
        eval_result = algo.evaluate()

        if last_entropy:
            print(
                "Policy entropy (max=ln(9)~2.20): "
                + ", ".join(f"{p}={e:.3f}" for p, e in sorted(last_entropy.items()))
            )

        # Report to Tune so ASHA can prune bad trials early
        if report_to_tune:
            metrics = summarize_evaluation(
                eval_result, n_evaders=n_evaders, time_limit=time_limit, beta=1, gamma=1
            )
            # global_step tracks total algo.train() calls across all stages.
            # ASHA reads this as time_attr so grace_period/max_t semantics
            # are in real iteration space, not tune.report() call count.
            metrics["algo_iteration"] = global_step
            # Per-policy mean action entropy — lands in result.json, the Tune
            # progress table, and TensorBoard (entropy_pursuer_policy /
            # entropy_evader_policy). See extract_entropies for how to read it.
            for pid, ent in last_entropy.items():
                metrics[f"entropy_{pid}"] = ent
            tune.report(metrics=metrics)

        # Save a checkpoint after each full stage (evader+pursuer training)
        if model_path and training == PURSUER:
            print(f"Saving stage {k + 1} at {model_path}/stage_{k + 1:05d}")
            model_name = model_path / f"stage_{k + 1:05d}"
            algo.save(str(model_name))

        # Alternate what policy gets trained
        train_evader = not train_evader

    return result


# ---------------------------------------------------------------------------
# Ray Tune hyperparameter search
# ---------------------------------------------------------------------------
def alternate_trainable(
    config: dict,
    n_stages: int = 4,
    iters_per_stage: int = 20,
    callbacks: list[type[RLlibCallback]] | None = None,
    # checkpoint_dir: Path | None = None,
) -> None:  # type: ignore[type-arg]
    """Trainable function for Ray Tune.

    Each trial builds an algo with sampled hyperparameters, runs the full
    alternating self-play loop, and reports metrics back to Tune after each
    stage half (i.e., every `iters_per_stage` algo.train() calls). With the
    defaults N_STAGES=8 and ITERS_PER_STAGE=10, that's 16 reports per
    fully-trained trial, enough for ASHA to prune underperforming trials.
    """
    # Pin this trial's process to single-threaded PyTorch (see Tune concurrency notes)
    torch.set_num_threads(1)  # pyright: ignore[reportPrivateImportUsage]
    try:
        torch.set_num_interop_threads(1)  # pyright: ignore[reportPrivateImportUsage]
    except RuntimeError:
        pass

    ppo_config = _build_ppo_config(
        config=config,
        callbacks=callbacks,
    )

    algo = ppo_config.build_algo()
    try:
        _run_alternating_loop(
            algo,
            n_stages=n_stages,
            iters_per_stage=iters_per_stage,
            report_to_tune=True,
            # checkpoint_dir=checkpoint_dir,
        )
    finally:
        algo.stop()


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def _set_policy_weights(algo, policy_id: str, weights: dict) -> None:
    """Set one policy's weights and propagate them to ALL runner groups.

    The previous approach — `algo.learner_group.set_weights(...)` followed by
    a sync gated on `num_env_runners > 0` — silently no-op'd on in-process
    configs (num_env_runners == 0): the learner copy was updated but the
    local env runner that actually collects rollouts never received the
    weights, and `algo.train()` only re-syncs the modules it just trained
    (the frozen policy is never among them). Net effect: opponent-pool
    sampling and the post-stage restore had zero influence on rollouts, and
    self-play silently degraded to always-vs-latest.

    `algo.set_weights(...)` routes through `Algorithm.set_state`, which sets
    the LearnerGroup state and then syncs weights to the training env
    runners (remote AND local) and the eval env runners. We verify the
    propagation reached the local rollout worker, mirroring the guard used
    in the pool-evaluation executor.
    """
    algo.set_weights({policy_id: weights})

    # Guard against the weights silently failing to reach the rollout
    # worker. Compare only the parameters both copies share: env-runner
    # modules can be inference-only and hold a subset of the learner params.
    runner_weights = algo.env_runner.get_weights([policy_id])[policy_id]
    common_keys = [k for k in runner_weights if k in weights]
    assert common_keys, (
        f"No overlapping parameter names between learner and env-runner "
        f"copies of {policy_id}; cannot verify weight propagation"
    )
    for k in common_keys:
        assert np.allclose(np.asarray(runner_weights[k]), np.asarray(weights[k])), (
            f"Weights for {policy_id} ({k}) did not propagate to the env "
            f"runner; rollouts would use a stale opponent"
        )


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
