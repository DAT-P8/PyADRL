from abc import abstractmethod
from .. import  GWActionResponse, SimStates
import numpy as np

class RewardFunction:
    @abstractmethod
    def calculate_rewards(response: GWActionResponse):
        pass


class GridWorldRewards(RewardFunction):
    # Rewards for evader
    REWARD_EVADER_TARGET_REACHED = 100
    REWARD_EVADER_CAUGHT = -100
    REWARD_EVADER_OUT_OF_BOUNDS = -1000
    REWARD_EVADER_MAX_TIMESTEPS = 50
    REWARD_EVADER_FAR_FROM_TARGET = -1  # Muiltiplier for distance to target
    REWARD_EVADER_FAR_FROM_PUSUERS = 1  # Multiplier for distance to closest pursuer

    # Rewards for pursuers
    REWARD_PURSUER_MAX_TIMESTEPS = -100  # Punish pursuers for not catching evader in time
    REWARD_PURSUER_TARGET_REACHED = -100  # Punish pursuers for letting evader reach target
    REWARD_PURSUER_CAUGHT_EVADER_SELF = 100  # Reward for catching the evader yourself
    REWARD_PURSUER_CAUGHT_EVADER_OTHERS = 10  # Reward for helping catch the evader
    REWARD_PURSUER_DESTROYED = -1000
    REWARD_PURSUER_FAR_FROM_EVADER = -1  # Multiplier for distance to evader

    def calculate_rewards(self, response, rewards, pursuers, evader, newly_destroyed, target_x, target_y):
        state = response.state

        # Punish pursuers that were destroyed
        for pursuer in newly_destroyed:
            rewards[pursuer.name] += self.REWARD_PURSUER_DESTROYED

        # Give rewards based on how the simulation terminated
        # If the simulation is still running this will just be skipped
        match state.simulation_state:
            case SimStates.EvadersCaptured:
                rewards[evader.name] += self.REWARD_EVADER_CAUGHT
                # reward only the pursuer that caught the evader
                for pursuer in pursuer:
                    if pursuer.x == evader.x and pursuer.y == evader.y:
                        rewards[pursuer.name] += self.REWARD_PURSUER_CAUGHT_EVADER_SELF
                    else:
                        rewards[pursuer.name] += self.REWARD_PURSUER_CAUGHT_EVADER_OTHERS
            case SimStates.EvadersCrashed:
                rewards[evader.name] += self.REWARD_EVADER_OUT_OF_BOUNDS
            case SimStates.TargetReached:
                rewards[evader.name] += self.REWARD_EVADER_TARGET_REACHED
                for pursuer in pursuer:
                    rewards[pursuer.name] += self.REWARD_PURSUER_TARGET_REACHED
            case SimStates.PursuersCrashed:
                pass
            case SimStates.TimeExceeded:
                rewards[self.evader.name] += self.REWARD_EVADER_MAX_TIMESTEPS
                for pursuer in pursuer:
                    rewards[pursuer.name] += self.REWARD_PURSUER_MAX_TIMESTEPS

        # punish the pursuers for being far from the evader to encourage them to move towards the evader
        for pursuer in pursuers:
            distance_to_evader = np.sqrt(
                (pursuer.x - evader.x) ** 2 + (pursuer.y - evader.y) ** 2
            )
            rewards[pursuer.name] += distance_to_evader * self.REWARD_PURSUER_FAR_FROM_EVADER

        # punish the evader for being far from the target to encourage it to move towards the target
        distance_to_target = np.sqrt(
            (evader.x - target_x) ** 2 + (evader.y - target_y) ** 2
        )
        rewards[evader.name] += distance_to_target * self.REWARD_EVADER_FAR_FROM_TARGET

        # reward the evader for being far from the pursuers to encourage it to move away from the pursuers
        closest_pursuer_distance = min(
            np.sqrt((pursuer.x - evader.x) ** 2 + (pursuer.y - evader.y) ** 2)
            for pursuer in pursuer
        )
        rewards[evader.name] += (
            closest_pursuer_distance * self.REWARD_EVADER_FAR_FROM_PUSUERS
        )

        return rewards
