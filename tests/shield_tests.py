from typing import Any
import numpy as np
import unittest

from numpy._typing import NDArray

from PyADRL.dtos.map_dtos import MapSpec, ObjectSpec, SquareMap, SquareObject
from PyADRL.dtos.ngw_dtos import Action, DroneAction, DroneState, State
from PyADRL.shielding.centralized_shield import CentralizedShield, Shield
from PyADRL.utils import utility

class ShieldingTests(unittest.TestCase):
    def test_object_shield(self):
        map = default_map()
        shield = CentralizedShield(map)

        drones: dict[int, tuple[DroneState, DroneAction, NDArray[Any]]] = {
            1: get_drone(1, 4, 4, Action(Action.ACTION_LEFT_DOWN), 2),
            2: get_drone(2, 2, 2, Action(Action.ACTION_RIGHT_UP), 2),
            3: get_drone(3, 7, 6, Action(Action.ACTION_UP), 2),
            4: get_drone(4, 8, 8, Action(Action.ACTION_UP), 2),
            5: get_drone(5, 2, 3, Action(Action.ACTION_RIGHT_UP), 1),
        }


        invalid_actions = shield.objects_collision_shield(drones)

        self.assertIn(1, invalid_actions)
        self.assertIn(2, invalid_actions)
        self.assertIn(3, invalid_actions)
        self.assertNotIn(4, invalid_actions)
        self.assertNotIn(5, invalid_actions)


    def test_out_of_bounds_shield(self):
        map = default_map()
        shield = CentralizedShield(map)

        drones: dict[int, tuple[DroneState, DroneAction, NDArray[Any]]] = {
            1: get_drone(1, 0, 0, Action(Action.ACTION_UP), 2),
            2: get_drone(2, 0, 0, Action(Action.ACTION_DOWN), 2),
            3: get_drone(3, 0, 0, Action(Action.ACTION_LEFT), 2),
            4: get_drone(4, 5, 5, Action(Action.ACTION_LEFT), 10),
            5: get_drone(5, 9, 9, Action(Action.ACTION_RIGHT), 1),
            6: get_drone(6, 8, 9, Action(Action.ACTION_RIGHT), 1),
        }

        invalid_actions = shield.out_of_bounds_shield(drones)

        self.assertNotIn(1, invalid_actions)
        self.assertIn(2, invalid_actions)
        self.assertIn(3, invalid_actions)
        self.assertIn(4, invalid_actions)
        self.assertIn(5, invalid_actions)
        self.assertNotIn(6, invalid_actions)


    def test_drone_collision_shield(self):
        map = default_map()
        shield = CentralizedShield(map)
        
        drones: dict[int, tuple[DroneState, DroneAction, NDArray[Any]]] = {
            1: get_drone(1, 1, 0, Action(Action.ACTION_LEFT), 1, is_evader=False),
            2: get_drone(2, 0, 0, Action(Action.ACTION_RIGHT), 1, is_evader=False),

            3: get_drone(3, 4, 2, Action(Action.ACTION_RIGHT_UP), 2, is_evader=False),
            4: get_drone(4, 6, 2, Action(Action.ACTION_LEFT_UP), 2, is_evader=True),
        }

        pair_wise_collisions = shield.drone_collisions_shield(drones)
        invalid_actions = [id for id1, id2 in pair_wise_collisions for id in [id1, id2]]

        self.assertIn(1, invalid_actions)
        self.assertIn(2, invalid_actions)
        self.assertNotIn(3, invalid_actions)
        self.assertNotIn(4, invalid_actions)


    def test_oob_shield(self):
        map = default_map()
        shield = CentralizedShield(map, rand_seed=100)

        # Two drones just yeeting off the map
        drone_states: list[DroneState] = [
            DroneState(1, 0, 0),
            DroneState(2, 9, 9),
        ]
        state = State(0, False, drone_states, [])

        drone_actions: list[DroneAction] = [
            DroneAction(1, 2, Action(Action.ACTION_LEFT_DOWN)), 
            DroneAction(2, 2, Action(Action.ACTION_RIGHT_UP)),
        ]

        safe_actions, _ = shield.shield(drone_actions, state)

        assert_safe_actions(safe_actions, state, map)


    def test_assert_fail(self):
        map = default_map()
        drone_states: list[DroneState] = [
            DroneState(1, 5, 3),
            DroneState(2, 6, 3),
        ]
        state = State(0, False, drone_states, [])

        drone_actions: list[DroneAction] = [
            DroneAction(1, 2, Action(Action.ACTION_RIGHT)), 
            DroneAction(2, 2, Action(Action.ACTION_LEFT)),
        ]
        
        try:
            assert_safe_actions(drone_actions, state, map)
        except:
            return

        raise Exception("assertion did not fail, as expected!")


    def test_oc_shield(self):
        map = default_map()
        shield = CentralizedShield(map, rand_seed=100)

        # Two drones just yeeting into an object
        # default map has objects at 3, 3 and 7, 7
        drone_states: list[DroneState] = [
            DroneState(1, 3, 4),
            DroneState(2, 7, 8),
        ]
        state = State(0, False, drone_states, [])

        drone_actions: list[DroneAction] = [
            DroneAction(1, 2, Action(Action.ACTION_DOWN)), 
            DroneAction(2, 2, Action(Action.ACTION_DOWN)),
        ]
        safe_actions, _ = shield.shield(drone_actions, state)
        assert_safe_actions(safe_actions, state, map)


    def test_dc_shield(self):
        map = default_map()
        shield = CentralizedShield(map, rand_seed=100)

        # Two drones just yeeting into each other
        # Also they are in the corner, so harder to find good action
        drone_states: list[DroneState] = [
            DroneState(1, 9, 8),
            DroneState(2, 9, 9),
        ]
        state = State(0, False, drone_states, [])

        drone_actions: list[DroneAction] = [
            DroneAction(1, 2, Action(Action.ACTION_UP)), 
            DroneAction(2, 2, Action(Action.ACTION_DOWN)),
        ]
        safe_actions, _ = shield.shield(drone_actions, state)
        assert_safe_actions(safe_actions, state, map)


    def test_shield_alt_state_dc(self):
        map = default_map()
        shield = CentralizedShield(map, rand_seed=100)

        # Two drones just yeeting into each other
        # Also they are in the corner, so harder to find good action
        drone_states: list[DroneState] = [
            DroneState(1, 0, 0),
            DroneState(2, 1, 0),
        ]
        state = State(0, False, drone_states, [])

        drone_actions: list[DroneAction] = [
            DroneAction(1, 1, Action(Action.ACTION_RIGHT)), 
            DroneAction(2, 1, Action(Action.ACTION_LEFT)), 
        ]
        _, state = shield.shield(drone_actions, state)

        if state == None:
            raise Exception("expected to find state, not None")
        if len(state.events) != 1:
            raise Exception(f"Expected exactly 1 event, instead got {len(state.events)}")
        for e in state.events:
            if e.collision_event == None:
                raise Exception("Expected a collision event")
            if 1 not in e.collision_event.drone_ids:
                raise Exception("Expected drone id 1 in out of bounds event")
            if 2 not in e.collision_event.drone_ids:
                raise Exception("Expected drone id 2 in out of bounds event")


    def test_shield_alt_state_oc(self):
        map = default_map()
        shield = CentralizedShield(map, rand_seed=100)

        # Two drones just yeeting into each other
        # Also they are in the corner, so harder to find good action
        drone_states: list[DroneState] = [
            DroneState(1, 3, 2),
        ]
        state = State(0, False, drone_states, [])

        drone_actions: list[DroneAction] = [
            DroneAction(1, 1, Action(Action.ACTION_UP)), 
        ]
        _, state = shield.shield(drone_actions, state)

        if state == None:
            raise Exception("expected to find state, not None")
        if len(state.events) != 1:
            raise Exception(f"Expected exactly 1 event, instead got {len(state.events)}")
        for e in state.events:
            if e.drone_object_collision_event == None:
                raise Exception("Expected a object collision event")
            if 1 not in e.drone_object_collision_event.drone_ids:
                raise Exception("Expected drone id 1 in out of bounds event")


    def test_shield_alt_state(self):
        map = default_map()
        shield = CentralizedShield(map, rand_seed=100)

        # Two drones just yeeting into each other
        # Also they are in the corner, so harder to find good action
        drone_states: list[DroneState] = [
            DroneState(1, 0, 0),
        ]
        state = State(0, False, drone_states, [])

        drone_actions: list[DroneAction] = [
            DroneAction(1, 2, Action(Action.ACTION_DOWN)), 
        ]
        _, state = shield.shield(drone_actions, state)

        if state == None:
            raise Exception("expected to find state, not None")
        if len(state.events) != 1:
            raise Exception(f"Expected exactly 1 event, instead got {len(state.events)}")
        for e in state.events:
            if e.out_of_bounds_event == None:
                raise Exception("Expected an out of bounds event")
            if 1 not in e.out_of_bounds_event.drone_ids:
                raise Exception("Expected drone id 1 in out of bounds event")


def assert_safe_actions(actions: list[DroneAction], state: State, map: MapSpec):
    xb, yb = (0, 0)
    if map.square_map != None:
        xb, yb = map.square_map.width, map.square_map.height
    else:
        raise Exception("did not recognize map type")

    object_positions: list[NDArray[Any]] = []
    if map.square_map != None:
        for obj in map.square_map.objects:
            if obj.square_object != None:
                object_positions.append(np.array([obj.square_object.x, obj.square_object.y]))

    new_positions: dict[int, tuple[NDArray[Any], NDArray[Any]]] = {}
    for drone in state.drone_states:
        action: DroneAction = DroneAction(drone.id, 0, Action(Action.ACTION_NOTHING))
        for a in actions:
            if a.id != drone.id:
                continue
            action = a
            break
        current_pos = np.array([drone.x, drone.y])
        new_pos = current_pos + utility.action_to_vector(action.action) * action.velocity
        new_positions[drone.id] = current_pos, new_pos

    for id in new_positions:
        curr, new_position = new_positions[id]
        if new_position[0] < 0  or xb <= new_position[0] or new_position[1] < 0 or yb <= new_position[1]:
            raise Exception(f"This position was out of bounds: ({new_position[0]}, {new_position[1]})")

        for obj_pos in object_positions:
            point = utility.sweep_pair(curr, new_position, obj_pos, obj_pos)
            if point.dot(point) < 0.001:
                raise Exception("Collision with object!")

        for other_id in new_positions:
            if id == other_id:
                continue
            a2, b2 = new_positions[other_id]
            point = utility.sweep_pair(a2, b2, curr, new_position)
            if point.dot(point) < .7:
                raise Exception("Collision between drones!")


def get_drone(
    id: int,
    x: int,
    y: int,
    action: Action,
    velocity: int,
    is_evader: bool = False,
    is_destroyed: bool = False
    ) -> tuple[DroneState, DroneAction, NDArray[Any]]:

    drone_state = DroneState(id, x, y, is_destroyed, is_evader)
    drone_action = DroneAction(id, velocity, action)
    new_pos = utility.action_to_vector(action) * velocity + np.array([drone_state.x, drone_state.y])
    return (drone_state, drone_action, new_pos)


def default_map() -> MapSpec:
    objects: list[ObjectSpec] = [
        ObjectSpec(square_object=SquareObject(3, 3)),
        ObjectSpec(square_object=SquareObject(7, 7)),
    ]

    return MapSpec(square_map=SquareMap(10, 10, 5, 5, objects))
