from ..dtos.ngw_dtos import Action
from ..dtos.ngw_dtos import (
    DroneAction,
)
from ..envs.ngw_drone import NGW_Drone


def get_direction(action: float) -> Action:
    if action == 0:
        return Action(Action.ACTION_NOTHING)

    action = (int(action) - 1) % 8
    match action:
        case 0:
            return Action(Action.ACTION_LEFT)
        case 1:
            return Action(Action.ACTION_LEFT_UP)
        case 2:
            return Action(Action.ACTION_UP)
        case 3:
            return Action(Action.ACTION_RIGHT_UP)
        case 4:
            return Action(Action.ACTION_RIGHT)
        case 5:
            return Action(Action.ACTION_RIGHT_DOWN)
        case 6:
            return Action(Action.ACTION_DOWN)
        case 7:
            return Action(Action.ACTION_LEFT_DOWN)
        case e:
            raise ValueError(f"Invalid action: {e}")


def get_velocity(action: float) -> int:
    # subtract 1 to account for the nothing action, then divide by 8 to get the velocity
    # add 1 back to get the velocity in the range [1, drone_velocity]
    return ((int(action) - 1) // 8) + 1


def get_drone_action(action: float, drone: NGW_Drone) -> DroneAction:
    return DroneAction(
        id=drone.id,
        action=get_direction(action),
        velocity=get_velocity(action),
    )
