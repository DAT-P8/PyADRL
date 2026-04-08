import ray
import tune
import grpc
from ray.tune import Tuner
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ..envs.gridworld_env import GridWorldEnvironment
from ray.rllib.algorithms.ppo.ppo import PPOConfig

from ..trainables.iterative_training import IterativeTraining


def gridworld_train():
    ray.shutdown()
    ray.init(log_to_drive=False)

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
            num_learners=2,
        )
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=5,
        )
        .training(
            train_batch_size=10000,
            minibatch_size=512,
            num_epochs=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .evaluation(evaluation_num_env_runners=0)
    )

    trainer = tune.with_parameters(IterativeTraining, test_string="test")
    stopping_criteria = {"training_iteration": 10}
    tuner = Tuner(
        trainer,
        param_space=config,
        run_config=tune.RunConfig(
            stop=stopping_criteria,
        ),
    )

    result = tuner.fit()
    print(result)
