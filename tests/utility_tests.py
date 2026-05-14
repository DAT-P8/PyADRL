import unittest
import math
import numpy as np

from PyADRL.dtos.ngw_dtos import Action
from PyADRL.utils import utility

class UtilityUnitTests(unittest.TestCase):
    def test_action_to_vector(self):
        cases: list[tuple[Action, int, int]] = [
            (Action(Action.ACTION_DOWN), 0, -1),
            (Action(Action.ACTION_LEFT_UP), -1, 1),
            (Action(Action.ACTION_NOTHING), 0, 0),
        ]

        for action, x, y in cases:
            arr = utility.action_to_vector(action)
            self.assertEqual(arr[0], x)
            self.assertEqual(arr[1], y)

    def test_sweep_pair(self):
        cases = [
            ## These 4 tests make sure a drone can pass through
            ## origin diagonally when objects are on left, right, up and down side of origin
            ( # close to the right
                np.array([1, 0]),
                np.array([1, 0]),
                np.array([-1, -1]),
                np.array([1, 1]),
                math.sqrt(0.5),
            ),
            ( # close to the left
                np.array([-1, 0]),
                np.array([-1, 0]),
                np.array([-1, -1]),
                np.array([1, 1]),
                math.sqrt(0.5),
            ),
            ( # close above origo
                np.array([0, 1]),
                np.array([0, 1]),
                np.array([-1, -1]),
                np.array([1, 1]),
                math.sqrt(0.5),
            ),
            ( # close below origo
                np.array([0, -1]),
                np.array([0, -1]),
                np.array([-1, -1]),
                np.array([1, 1]),
                math.sqrt(0.5),
            ),

            ### Direct collision of two drones should have a distance to point very low
            (
                np.array([1, 1]),
                np.array([-1, -1]),
                np.array([-1, -1]),
                np.array([1, 1]),
                math.sqrt(1e-5),
            )
        ]

        for b1, a1, b2, a2, max_distance in cases:
            point = utility.sweep_pair(b1, a1, b2, a2)
            p_dist: float = point.dot(point)
            self.assertLessEqual(p_dist, max_distance)
