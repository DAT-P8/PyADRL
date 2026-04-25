from ...dtos.ngw_dtos import State
from .rewards import RewardFunction
from ..ngw_drone import NGW_Drone
from ..map_configs.map_config import MapConfig
from ...utils.chebeshyv import chebyshev_distance


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

        target_reached_set: set[int] = set()
        out_of_bounds_set: set[int] = set()
        pursuer_entered_target_set: set[int] = set()
        drone_object_collision_set: set[int] = set()
        # collision_dict is mapping drone -> to all drones it collides with, includes self
        collision_dict: dict[int, set[int]] = {}
        evaders_caught: set[int] = set()

        for event in new_state.events:
            if event.drone_object_collision_event is not None:
                drone_object_collision_set.update(
                    event.drone_object_collision_event.drone_ids
                )
            elif event.out_of_bounds_event is not None:
                out_of_bounds_set.update(event.out_of_bounds_event.drone_ids)
            elif event.pursuer_entered_target_event is not None:
                pursuer_entered_target_set.update(
                    event.pursuer_entered_target_event.drone_ids
                )
            elif event.target_reached_event is not None:
                target_reached_set.update(event.target_reached_event.drone_ids)
            elif event.collision_event is not None:
                for id_key in event.collision_event.drone_ids:
                    if id_key in collision_dict:
                        collision_dict[id_key].update(event.collision_event.drone_ids)
                    else:
                        collision_dict[id_key] = set(event.collision_event.drone_ids)
            else:
                # If we had logging done, then this would be a warning
                pass

        for id in target_reached_set:
            drone = all_drones[id]
            if drone.is_evader:
                rewards[drone.name] += self.REWARD_EVADER_TARGET_REACHED

        for id in out_of_bounds_set:
            drone = all_drones[id]
            if drone.is_evader:
                rewards[drone.name] += self.REWARD_EVADER_OUT_OF_BOUNDS
            else:
                rewards[drone.name] += self.REWARD_PURSUER_OUT_OF_BOUNDS

        for id in collision_dict:
            collision_set = collision_dict[id]

            drone = all_drones[id]
            evaders = [x for x in collision_set if all_drones[x].is_evader]
            pursuers = [x for x in collision_set if not all_drones[x].is_evader]

            # evaders and pursuers present in collision, it was a capture
            if len(evaders) != 0 and len(pursuers) != 0:
                evaders_caught.update(evaders)

                if drone.is_evader:
                    rewards[drone.name] += self.REWARD_EVADER_CAUGHT
                else:
                    rewards[drone.name] += (
                        len(evaders) * self.REWARD_PURSUER_CAUGHT_EVADER_SELF
                    )

            else:
                if drone.is_evader:
                    rewards[drone.name] += self.REWARD_EVADER_DESTROYED
                else:
                    rewards[drone.name] += self.REWARD_PURSUER_DESTROYED

        for id in pursuer_entered_target_set:
            pass

        for id in drone_object_collision_set:
            pass

        for drone in all_drones.values():
            if drone.is_evader:
                # reward the evader for being far from the pursuers to encourage it to move away from the pursuers
                distance_to_pursuer = min(
                    (
                        chebyshev_distance(drone.x, drone.y, pursuer.x, pursuer.y)
                        for pursuer in drones["pursuers"]
                        if not pursuer.destroyed
                    ),
                    default=0,
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
                    (
                        chebyshev_distance(drone.x, drone.y, evader.x, evader.y)
                        for evader in drones["evaders"]
                        if not evader.destroyed
                    ),
                    default=0,
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
