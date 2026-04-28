from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec


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
    callbacks: list[RLlibCallback] = [],
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
        .evaluation(evaluation_num_env_runners=0)
        # .callbacks(callbacks_class = callbacks)
    )
