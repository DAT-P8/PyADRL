from .rewards import RewardFunction
from ..ngw_drone import NGW_Drone
from ..map_configs.map_config import MapConfig
from ...utils.ngw2d_client import State, EventTypes
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
    REWARD_EVADER_COLLISION_OBJECT = -200

    # Rewards for pursuers
    REWARD_PURSUER_MAX_TIMESTEPS = (
        -50
    )  # Punish pursuers for not catching evader in time
    REWARD_PURSUER_TARGET_REACHED = (
        -50
    )  # Punish pursuers for letting evader reach target
    REWARD_PURSUER_CAUGHT_EVADER_SELF = 100  # Rew ard for catching the evader yourself
    REWARD_PURSUER_CAUGHT_EVADER_OTHERS = 20  # Reward for helping catch the evader
    REWARD_PURSUER_DESTROYED = -1000
    REWARD_PURSUER_FAR_FROM_EVADER = -1  # Multiplier for distance to evader
    REWARD_PURSUER_OUT_OF_BOUNDS = -20
    REWARD_PURSUER_COLLISION_OBJECT = -200

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
        for event_type, event_occurances in new_state.events.items():
            flat_drone_ids = []
            for drone_ids in event_occurances:
                for id in drone_ids:
                    if id not in all_drones:
                        raise ValueError(
                            f"Terminated agent with id {id} recieved from {event_type}"
                        )
                    flat_drone_ids.append(id)

            match event_type:
                case EventTypes.TargetReachedEvent:
                    for id in flat_drone_ids:
                        d = all_drones[id]
                        rewards[d.name] += self.REWARD_EVADER_TARGET_REACHED
                case EventTypes.OutOfBoundsEvent:
                    for id in flat_drone_ids:
                        d = all_drones[id]
                        if d.is_evader:
                            rewards[d.name] += self.REWARD_EVADER_OUT_OF_BOUNDS
                        else:
                            rewards[d.name] += self.REWARD_PURSUER_OUT_OF_BOUNDS
                case EventTypes.CollisionEvent:
                    for id in flat_drone_ids:
                        d = all_drones[id]
                        if d.is_evader:
                            rewards[d.name] += self.REWARD_EVADER_DESTROYED
                        else:
                            rewards[d.name] += self.REWARD_PURSUER_DESTROYED
                case EventTypes.CaptureEvent:
                    for id in flat_drone_ids:
                        d = all_drones[id]
                        if d.is_evader:
                            rewards[d.name] += self.REWARD_EVADER_CAUGHT
                        else:
                            rewards[d.name] += self.REWARD_PURSUER_CAUGHT_EVADER_SELF
                    other_pursuers = [
                        d
                        for d in drones["pursuers"]
                        if d.id not in flat_drone_ids and not d.destroyed
                    ]
                    for p in other_pursuers:
                        rewards[p.name] += self.REWARD_PURSUER_CAUGHT_EVADER_OTHERS
                case EventTypes.PursuerEnteredTargetEvent:
                    for id in flat_drone_ids:
                        d = all_drones[id]
                        rewards[d.name] += self.REWARD_PURSUER_TARGET_REACHED
                case EventTypes.DroneObjectCollisionEvent:
                    print(f"Collision event with drones")
                    for id in flat_drone_ids:
                        d = all_drones[id]
                        if d.is_evader:
                            rewards[d.name] += self.REWARD_EVADER_COLLISION_OBJECT
                        else:
                            rewards[d.name] += self.REWARD_PURSUER_COLLISION_OBJECT
                case _:
                    raise ValueError(f"Recieved unkown event type {event_type}")

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
