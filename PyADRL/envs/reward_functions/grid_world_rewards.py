from .rewards import RewardFunction
import numpy as np

class GridWorldRewards(RewardFunction):
    # Rewards for evader
    REWARD_EVADER_TARGET_REACHED = 100
    REWARD_EVADER_CAUGHT = -100
    REWARD_EVADER_OUT_OF_BOUNDS = -1000
    REWARD_EVADER_MAX_TIMESTEPS = 50
    REWARD_EVADER_FAR_FROM_TARGET = -1  # Muiltiplier for distance to target
    REWARD_EVADER_FAR_FROM_PUSUERS = 1  # Multiplier for distance to closest pursuer

    # Rewards for pursuers
    REWARD_PURSUER_MAX_TIMESTEPS = -100  # Punish pursuers for not catching evader in time
    REWARD_PURSUER_TARGET_REACHED = -100  # Punish pursuers for letting evader reach target
    REWARD_PURSUER_CAUGHT_EVADER_SELF = 100  # Reward for catching the evader yourself
    REWARD_PURSUER_CAUGHT_EVADER_OTHERS = 10  # Reward for helping catch the evader
    REWARD_PURSUER_DESTROYED = -1000
    REWARD_PURSUER_FAR_FROM_EVADER = -1  # Multiplier for distance to evader

    # TODO: Fix arguements
    def calculate_rewards(self, new_state, agents, max_time, pursuers, evaders, target_x, target_y):
        rewards: dict[str, float] = {a: 0 for a in self.agents}
        all_drones = {d.id: d for d in self.evaders + self.pursuers}

        # distribute rewards for events that occured
        for event in new_state.events:
            which_one = event.WhichOneof("event_case")
            match which_one:
                case "target_reached":
                    id = event.target_reached.drone_id
                    d = all_drones[id]
                    if d is None:
                        raise Exception(f"Drone with id {id} not found")
                    rewards[d.name] += self.REWARD_EVADER_TARGET_REACHED
                case "drone_crashed":
                    id = event.drone_out_of_bounds.drone_id
                    d = all_drones[id]
                    if d is None:
                        raise Exception(f"Drone with id {id} not found")
                    if d.is_evader:
                        rewards[d.name] += self.REWARD_EVADER_OUT_OF_BOUNDS
                    else:
                        rewards[d.name] += self.REWARD_PURSUER_OUT_OF_BOUNDS
                case "collision":
                    ids = [d for d in event.collision.drone_ids]
                    for id in ids:
                        d = all_drones[id]
                        if d is None:
                            raise Exception(f"Drone with id {id} not found")
                        if d.is_evader:
                            rewards[d.name] += self.REWARD_EVADER_DESTROYED
                        else:
                            rewards[d.name] += self.REWARD_PURSUER_DESTROYED

        # punish the pursuers for being far from the evader to encourage them to move towards the evaderforfor
        for pursuer in pursuers:
            distance_to_evader = min(np.sqrt(
                (pursuer.x - evader.x) ** 2 + (pursuer.y - evader.y) ** 2)
                for evader in evaders
            )
            rewards[pursuer.name] += distance_to_evader * self.REWARD_PURSUER_FAR_FROM_EVADER
    
        # punish the evader for being far from the target to encourage it to move towards the target
        for evader in evaders:
            distance_to_target = np.sqrt(
                (evader.x - target_x) ** 2 + (evader.y - target_y) ** 2
            )
            rewards[evader.name] += distance_to_target * self.REWARD_EVADER_FAR_FROM_TARGET
    
        # reward the evader for being far from the pursuers to encourage it to move away from the pursuers
        closest_pursuer_distance = min(
            np.sqrt((pursuer.x - evader.x) ** 2 + (pursuer.y - evader.y) ** 2)
            for pursuer in pursuer
        )

        rewards[evader.name] += (
            closest_pursuer_distance * self.REWARD_EVADER_FAR_FROM_PUSUERS
        )

        if max_time:
            for e in self.evaders:
                rewards[e.name] += self.REWARD_EVADER_MAX_TIMESTEPS
            for pursuer in self.pursuers:
                rewards[pursuer.name] += self.REWARD_PURSUER_MAX_TIMESTEPS

        return rewards
