from abc import ABCMeta, abstractmethod
import random
from typing import Any, override
import numpy as np
from numpy._typing import NDArray
from PyADRL.dtos.map_dtos import MapSpec
from PyADRL.shielding.shield import Shield
from PyADRL.utils import utility
from ..dtos.ngw_dtos import (
    Action,
    CollisionEvent,
    DroneAction,
    DroneObjectCollisionEvent,
    DroneState,
    Event,
    OutOfBoundsEvent,
    State,
)


class CentralizedShield(Shield):
    def __init__(
        self,
        map: MapSpec,
        max_velocity: int = 1,
        threshold: float = 1e-4,
        threshold_drone_collisions: float = 0.7,
        rand_seed: int = 42,
        max_depth: int = 1000,
    ) -> None:
        super().__init__()
        self.map = map
        self.threshold = threshold
        self.random = random.Random(rand_seed)
        self.max_velocity = max_velocity
        self.max_depth = max_depth

        # this is the magic number that should not collide
        # when crossing diagonally, but collide otherwise
        # found by 0.5^2 + 0.5^2 = x^2
        # where 0.5 is half the length of a a square
        self.threshold_drone_collisions = threshold_drone_collisions

    # equiprobable random drone actions
    def get_random_drone_action(self, id: int) -> DroneAction:
        # all permutations off non-zero velocity actions
        dir_perms = 8 * self.max_velocity
        # +1 for the nothing action
        all_actions_n = dir_perms + 1

        # -1 because the range is inclusive of upper bound
        action_raw = self.random.randint(0, all_actions_n - 1)

        if action_raw == 0:
            return DroneAction(id, 0, Action(Action.ACTION_NOTHING))

        action_raw -= 1
        # we can view action_raw as the idx of a 2D array laid out in 1D
        row_idx = action_raw // 8  # row idx is the velocity
        col_idx = action_raw % 8  # col idx is the actual action value

        # non-nothing action values go from 2-9 and col idx is 0-7
        action = Action(col_idx + 2)

        return DroneAction(id, row_idx + 1, action)

    def out_of_bounds_shield(
        self, drones: dict[int, tuple[DroneState, DroneAction, NDArray[Any]]]
    ) -> set[int]:
        bounds: tuple[int, int] | None = None
        if self.map.square_map != None:
            y = self.map.square_map.height
            x = self.map.square_map.width
            bounds = (x, y)

        if bounds == None:
            raise Exception("Could not find bounds")

        invalid_actions: set[int] = set()
        xb, yb = bounds
        for id in drones:
            _, _, new_position = drones[id]
            if (
                0 <= new_position[0]
                and new_position[0] < xb
                and 0 <= new_position[1]
                and new_position[1] < yb
            ):
                continue
            invalid_actions.add(id)

        return invalid_actions

    def drone_collisions_shield(
        self, drones: dict[int, tuple[DroneState, DroneAction, NDArray[Any]]]
    ) -> set[tuple[int, int]]:
        invalid_actions: set[tuple[int, int]] = set()

        seen_keys: set[int] = set()
        for id1 in drones:
            seen_keys.add(id1)
            d1_state, _, d1_after = drones[id1]
            d1_before = np.array([d1_state.x, d1_state.y])

            for id2 in drones:
                if id2 in seen_keys:
                    continue

                d2_state, _, d2_after = drones[id2]
                if d1_state.is_evader != d2_state.is_evader:
                    continue
                d2_before = np.array([d2_state.x, d2_state.y])

                distance_vec = utility.sweep_pair(
                    d1_before, d1_after, d2_before, d2_after
                )
                distance_sq: float = distance_vec.dot(distance_vec)
                if distance_sq > self.threshold_drone_collisions:
                    continue

                invalid_actions.add((id1, id2))

        return invalid_actions

    def objects_collision_shield(
        self, drones: dict[int, tuple[DroneState, DroneAction, NDArray[Any]]]
    ) -> set[int]:
        object_positions: list[NDArray[Any]] = []

        if self.map.square_map != None:
            for o in self.map.square_map.objects:
                if o.square_object != None:
                    object_positions.append(
                        np.array([o.square_object.x, o.square_object.y])
                    )

        invalid_actions: set[int] = set()
        for id in drones:
            drone_state, _, drone_after = drones[id]
            drone_before = np.array([drone_state.x, drone_state.y])

            for obj in object_positions:
                dist_to_point_vec = utility.sweep_pair(
                    drone_before, drone_after, obj, obj
                )
                distance_sq: float = dist_to_point_vec.dot(dist_to_point_vec)

                if distance_sq > self.threshold:
                    continue

                invalid_actions.add(id)
                break

        return invalid_actions

    def get_drone_state_after_action(
        self, state: DroneState, action: DroneAction
    ) -> DroneState:
        new_pos = utility.action_to_vector(action.action) * action.velocity + np.array(
            [state.x, state.y]
        )
        return DroneState(
            state.id, new_pos[0], new_pos[1], state.destroyed, state.is_evader
        )

    @override
    def shield(
        self, actions: list[DroneAction], state: State
    ) -> tuple[list[DroneAction], State | None]:
        drone_actions: dict[int, tuple[DroneState, DroneAction, NDArray[Any]]] = {}

        for drone in state.drone_states:
            action = DroneAction(drone.id, 0, Action(Action.ACTION_NOTHING))
            for a in actions:
                if a.id != drone.id:
                    continue
                action = a
                break
            new_pos = utility.action_to_vector(
                action.action
            ) * action.velocity + np.array([drone.x, drone.y])
            drone_actions[drone.id] = (drone, action, new_pos)

        oob_ids = self.out_of_bounds_shield(drone_actions)
        oc_ids = self.objects_collision_shield(drone_actions)
        dc_pairs = self.drone_collisions_shield(drone_actions)
        dc_ids: set[int] = set([id for id1, id2 in dc_pairs for id in [id1, id2]])

        # Create an alternative state, which is the state that would have resulted from taking the initial actions
        destroyed_ids = oob_ids.union(oc_ids).union(dc_ids)
        alt_es: list[Event] = []
        alt_ds: list[DroneState] = [
            self.get_drone_state_after_action(state, action)
            for state, action, _ in drone_actions.values()
        ]
        for ds in alt_ds:
            if ds.id in destroyed_ids:
                ds.destroyed = True

        if len(oc_ids) != 0:
            alt_es.append(
                Event(
                    drone_object_collision_event=DroneObjectCollisionEvent(
                        [id for id in oc_ids]
                    )
                )
            )

        out_of_bounds_ids_filtered = [id for id in oob_ids if id not in oc_ids]
        if len(out_of_bounds_ids_filtered) != 0:
            alt_es.append(
                Event(out_of_bounds_event=OutOfBoundsEvent(out_of_bounds_ids_filtered))
            )

        for d1, d2 in dc_pairs:
            alt_es.append(Event(collision_event=CollisionEvent([d1, d2])))

        actions_to_replace = oc_ids.union(dc_ids.union(oob_ids))
        depth = 0
        while len(actions_to_replace) != 0 and depth < self.max_depth:
            depth += 1
            for id in actions_to_replace:
                s, _, _ = drone_actions[id]
                action = self.get_random_drone_action(id)
                new_pos = utility.action_to_vector(
                    action.action
                ) * action.velocity + np.array([s.x, s.y])
                drone_actions[id] = s, action, new_pos

            oob_ids = self.out_of_bounds_shield(drone_actions)
            oc_ids = self.objects_collision_shield(drone_actions)
            dc_pairs = self.drone_collisions_shield(drone_actions)
            dc_ids = set([id for id1, id2 in dc_pairs for id in [id1, id2]])
            actions_to_replace = oc_ids.union(dc_ids.union(oob_ids))

        if len(actions_to_replace) == 0:
            alt_state = State(
                state.sim_id, state.terminated, alt_ds, alt_es, state.objects
            )
            return [action for _, action, _ in drone_actions.values()], alt_state

        raise Exception(f"Exceeded maximum iteration of {self.max_depth}")
