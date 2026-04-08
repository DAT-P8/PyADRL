from ..ngw.v1.ngw2d_pb2 import Action


def get_action(action: float) -> Action:
    match action:
        case 0:
            return Action.ACTION_NOTHING
        case 1:
            return Action.ACTION_LEFT
        case 2:
            return Action.ACTION_LEFT_UP
        case 3:
            return Action.ACTION_UP
        case 4:
            return Action.ACTION_RIGHT_UP
        case 5:
            return Action.ACTION_RIGHT
        case 6:
            return Action.ACTION_RIGHT_DOWN
        case 7:
            return Action.ACTION_DOWN
        case 8:
            return Action.ACTION_LEFT_DOWN
        case _:
            raise ValueError(f"Invalid action: {action}")
