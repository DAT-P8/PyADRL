from .rewards import RewardFunction
from ..ngw_drone import NGW_Drone
from ..map_configs.map_config import MapConfig
from ...ngw.v1.ngw2d_pb2 import State
import numpy as np


class GridWorldRewards(RewardFunction):
    # Rewards for evader
    REWARD_EVADER_TARGET_REACHED = 100
    REWARD_EVADER_CAUGHT = -100
    REWARD_EVADER_OUT_OF_BOUNDS = -1000
    REWARD_EVADER_MAX_TIMESTEPS = 50
    REWARD_EVADER_FAR_FROM_TARGET = -1  # Muiltiplier for distance to target
    REWARD_EVADER_FAR_FROM_PUSUERS = 1  # Multiplier for distance to closest pursuer
    REWARD_EVADER_DESTROYED = -100

    # Rewards for pursuers
    REWARD_PURSUER_MAX_TIMESTEPS = (
        -100
    )  # Punish pursuers for not catching evader in time
    REWARD_PURSUER_TARGET_REACHED = (
        -100
    )  # Punish pursuers for letting evader reach target
    REWARD_PURSUER_CAUGHT_EVADER_SELF = 100  # Reward for catching the evader yourself
    REWARD_PURSUER_CAUGHT_EVADER_OTHERS = 10  # Reward for helping catch the evader
    REWARD_PURSUER_DESTROYED = -1000
    REWARD_PURSUER_FAR_FROM_EVADER = -1  # Multiplier for distance to evader
    REWARD_PURSUER_OUT_OF_BOUNDS = -100

    def calculate_rewards(
        self,
        new_state: State,
        agents: list[str],
        drones: dict[str, list[NGW_Drone]],
        map_config: MapConfig,
        time_limit_reached: bool,
    ) -> dict[str, float]:
        rewards: dict[str, float] = {a: 0 for a in agents}
        all_drones = {
            d.id: d for d in drones["evaders"] + drones["pursuers"] if d.name in agents
        }

        # distribute rewards for events that occured
        for event in new_state.events:
            which_one = event.WhichOneof("event_oneof")
            match which_one:
                case "target_reached_event ":
                    ids = [d for d in event.target_reached_event.drone_ids]
                    for id in ids:
                        d = all_drones[id]
                        if d is None:
                            raise Exception(f"Drone with id {id} not found")
                        rewards[d.name] += self.REWARD_EVADER_TARGET_REACHED
                case "out_of_bounds_event ":
                    ids = [d for d in event.out_of_bounds_event.drone_ids]
                    for id in ids:
                        d = all_drones[id]
                        if d is None:
                            raise Exception(f"Drone with id {id} not found")
                        if d.is_evader:
                            rewards[d.name] += self.REWARD_EVADER_OUT_OF_BOUNDS
                        else:
                            rewards[d.name] += self.REWARD_PURSUER_OUT_OF_BOUNDS
                case "collision_event ":
                    ids = [d for d in event.collision_event.drone_ids]
                    for id in ids:
                        d = all_drones[id]
                        if d is None:
                            raise Exception(f"Drone with id {id} not found")
                        if d.is_evader:
                            rewards[d.name] += self.REWARD_EVADER_DESTROYED
                        else:
                            rewards[d.name] += self.REWARD_PURSUER_DESTROYED

        for drone in all_drones.values():
            if drone.is_evader:
                # reward the evader for being far from the pursuers to encourage it to move away from the pursuers
                distance_to_pursuer = min(
                    np.sqrt((drone.x - pursuer.x) ** 2 + (drone.y - pursuer.y) ** 2)
                    for pursuer in drones["pursuers"]
                )
                rewards[drone.name] += (
                    distance_to_pursuer * self.REWARD_EVADER_FAR_FROM_PUSUERS
                )

                # punish evaders for being far from the target to encourage them to move towards it
                distance_to_target = map_config.distance_to_target(drone.x, drone.y)
                rewards[drone.name] += (
                    distance_to_target * self.REWARD_EVADER_FAR_FROM_TARGET
                )
            else:
                # punish the pursuers for being far from the evader to encourage them to move towards the evader
                distance_to_evader = min(
                    np.sqrt((drone.x - evader.x) ** 2 + (drone.y - evader.y) ** 2)
                    for evader in drones["evaders"]
                )
                rewards[drone.name] += (
                    distance_to_evader * self.REWARD_PURSUER_FAR_FROM_EVADER
                )

        if time_limit_reached:
            for drone in all_drones.values():
                if drone.is_evader:
                    rewards[drone.name] += self.REWARD_EVADER_MAX_TIMESTEPS
                else:
                    rewards[drone.name] += self.REWARD_PURSUER_MAX_TIMESTEPS

        return rewards
