import numpy as np
from ..dtos.ngw_dtos import (
    Action
)

def action_to_vector(action: Action) -> np.ndarray:
    action_map: dict[int, np.ndarray] = {
        Action.ACTION_NOTHING:    np.array([0, 0]),
        Action.ACTION_RIGHT:      np.array([1, 0]),
        Action.ACTION_LEFT:       np.array([-1, 0]),
        Action.ACTION_UP:         np.array([0, 1]),
        Action.ACTION_DOWN:       np.array([0, -1]),
        Action.ACTION_RIGHT_UP:   np.array([1, 1]),
        Action.ACTION_LEFT_UP:    np.array([-1, 1]),
        Action.ACTION_RIGHT_DOWN: np.array([1, -1]),
        Action.ACTION_LEFT_DOWN:  np.array([-1, -1]),
    }
    vec = action_map.get(action.val)
    if vec is None:
        raise ValueError(f"Did not recognize action: {action.val}")
    
    return vec


def sweep_pair(
    before_v1: np.ndarray,
    after_v1: np.ndarray,
    before_v2: np.ndarray,
    after_v2: np.ndarray,
) -> np.ndarray:
    v1_mov = after_v1 - before_v1
    v2_mov = after_v2 - before_v2
    delta_mov = v1_mov - v2_mov

    P = np.zeros(3)
    A = before_v1 - before_v2
    B = A + delta_mov

    return project_point_onto_segment(P, A, B)

def project_point_onto_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ap = p - a

    ab_dot = np.dot(ab, ab)

    if ab_dot == 0.0:
        return a  # a and b are the same point

    t = np.dot(ap, ab) / ab_dot
    t = np.clip(t, 0.0, 1.0)

    return a + t * ab
