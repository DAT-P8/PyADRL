from PyADRL.ngw.v1.ngw2d_pb2 import ACTION_DOWN
from ..dtos.ngw_dtos import (
    Action,
    DroneAction,
    DroneState,
    State
)
from typing import override
from PyADRL.shielding.shield import Shield
import numpy as np
from ..utils.utils import (
    action_to_vector,
    sweep_pair
)

# This point can be very close to zero as collision with objects only happen
# in discrete space, e.g. object at x, y and drone at x, y  -> collision
THRESHOLD = 1e-3

class ObjectCollisionShield(Shield):
    def __init__(self, target_positions: list[tuple[int, int]]) -> None:
        super().__init__()
        self.targets = [np.array([x, y]) for x, y in target_positions]

    def get_action_directions(self) -> list[tuple[Action, np.ndarray]]:
        actions = [
            Action.ACTION_DOWN,
            Action.ACTION_LEFT_DOWN,
            Action.ACTION_LEFT,
            Action.ACTION_LEFT_UP,
            Action.ACTION_UP,
            Action.ACTION_RIGHT_UP,
            Action.ACTION_RIGHT,
            Action.ACTION_RIGHT_DOWN
        ]

        actions = [Action(x) for x in actions]

        # A collision WOULD happened with this target
        return [(x, np.linalg.norm(action_to_vector(x))) for x in actions]

    @override
    def shield(self, action: DroneAction, state: State) -> tuple[int, DroneAction]:
        action_vec = action_to_vector(action.action) * action.velocity

        drone: DroneState | None = None
        for d in state.drone_states:
            if d.id != action.id:
                continue
            drone = d
            break

        if drone is None:
            raise Exception("Did not find a drone for this action")

        pos_before = np.array([drone.x, drone.y])
        pos_after = pos_before + action_vec

        suggestions = []
        for target in self.targets:
            point = sweep_pair(pos_before, pos_after, target, target)
            if point > THRESHOLD:
                continue

            for (action, dir) in self.get_action_directions():
            # A collision WOULD happened with this target

        raise NotImplemented("shield has not been implemented")
