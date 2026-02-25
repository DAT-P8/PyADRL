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
