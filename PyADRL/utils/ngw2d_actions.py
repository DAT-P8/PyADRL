from ..ngw.v1.ngw2d_pb2 import Action


def get_action(action: float) -> Action:
    if action == 0:
        return Action.ACTION_NOTHING
    elif action == 1:
        return Action.ACTION_LEFT
    elif action == 2:
        return Action.ACTION_RIGHT
    elif action == 3:
        return Action.ACTION_UP
    elif action == 4:
        return Action.ACTION_DOWN
    else:
        raise ValueError(f"Invalid action: {action}")
