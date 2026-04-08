import random
from ray.tune import Trainable


class Alternate_Training(Trainable):
    def setup(self, checkpoint=None, p_old=0.3, n_stages=4, iters_per_stage=20):
        self.p_old = p_old
        self.n_stages = n_stages
        self.iters_per_stage = iters_per_stage

        self.iteration_cnt = 0
        self.pools = {"evader": [], "pursuer": []}
        if checkpoint is not None:
            print("Restoring checkpoint from checkpoint:", checkpoint)
            self.algo.restore(checkpoint)
            assert self.algo.learner_group is not None
            weights = self.algo.learner_group.get_weights()
            self.pools["pursuer"].append(weights["pursuer_policy"])
            self.pools["evader"].append(weights["evader_policy"])

    def step(self):
        for current, frozen in [("pursuer", "evader"), ("evader", "pursuer")]:
            print(f"\nStage {self.iteration_cnt + 1}: training {current}")

            assert self.algo.learner_group is not None
            assert self.algo.config is not None

            self.algo.config._is_frozen = False
            self.algo.learner_group.foreach_learner(
                lambda learner, *args: learner.config.multi_agent(
                    policies_to_train=[f"{current}_policy"]
                )
            )
            self.algo.config._is_frozen = True

            if self.pools[frozen]:
                opp_weights = self.sample_opponent(self.pools[frozen])
                self.algo.learner_group.set_weights({f"{frozen}_policy": opp_weights})

            for i in range(self.iters_per_stage):
                result = self.algo.train()
                # TODO: Add metrics
                # mean = result["env_runners"]["agent_episode_returns_mean"]
                # rewards.append(mean)
                # iteration_data = build_train_iteration_data(result, i + 1)
                # episodes_data.extend(iteration_data.get("episodes", []))
                # # Keep a live per-episode log while training is in progress.
                # write_metrics(train_metrics_path, {"episodes": episodes_data})
            assert self.algo.learner_group is not None
            updated_weights = self.algo.learner_group.get_weights()[
                f"{current}_policy"
            ]  # New weights?
            self.pools[current].append(updated_weights)

    def sample_opponent(self, pool: list[dict]) -> dict:
        """With prob P_OLD sample a random old policy, otherwise use the latest."""
        if len(pool) == 1:
            return pool[-1]  # most recent
        elif random.random() < self.p_old:
            return random.choice(pool[:-1])  # Sample from all but the last policy
        else:
            return pool[-1]
