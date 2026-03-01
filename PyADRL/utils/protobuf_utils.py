import random
import grid_world_pb2 as gwpb


def rand_action() -> gwpb.GWAction:
    action = random.choice(
        [
            gwpb.GWAction.NOTHING,
            gwpb.GWAction.LEFT,
            gwpb.GWAction.RIGHT,
            gwpb.GWAction.UP,
            gwpb.GWAction.DOWN,
        ]
    )
    return action

def get_action(action: int) -> gwpb.GWAction:
    if action == 0:
        return gwpb.GWAction.NOTHING
    elif action == 1:
        return gwpb.GWAction.LEFT
    elif action == 2:
        return gwpb.GWAction.RIGHT
    elif action == 3:
        return gwpb.GWAction.UP
    elif action == 4:
        return gwpb.GWAction.DOWN
    else:
        raise ValueError(f"Invalid action: {action}")