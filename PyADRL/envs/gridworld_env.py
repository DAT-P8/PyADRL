"""Gridworld MA area defense environment"""

from pettingzoo import ParallelEnv
import grpc
from .. import grid_world_pb2
from ..utils.protobuf_utils import get_action
from ..utils.gridworld_client import GridWorldClient
from gymnasium.spaces import Box, Discrete
import numpy as np
import time

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


class Drone:
    def __init__(self, id: int, x: int, y: int, is_evader: bool):
        self.id: int = id
        self.x = x
        self.y = y
        self.name = "evader" if is_evader else f"pursuer_{id}"
        self.is_evader = is_evader
        self.destroyed: bool = False


class GridWorldEnvironment(ParallelEnv):
    metadata = {
        "name": "gridworld_environment_v0",
    }

    def __init__(self, channel: grpc.Channel, step_delay: float = 0.0):
        self.id: int | None = None
        self.client = GridWorldClient(channel)
        self.step_delay = step_delay  # Delay to slow down steps for visualization
        self.target_x: int = 5
        self.target_y: int = 5
        self.pursuer: list[Drone] = []
        self.evader: Drone | None = None
        self.timestep: int = 0
        self.agents = []

    def _get_obs(self):
        if self.evader is None or len(self.pursuer) == 0:
            raise ValueError("Pursuer or evader not initialized")

        obs = []
        for p in self.pursuer:
            obs += [p.x / 10.0, p.y / 10.0]
        obs += [self.evader.x / 10.0, self.evader.y / 10.0]
        obs += [self.target_x / 10.0, self.target_y / 10.0]
        obs_array = np.array(obs, dtype=np.float32)

        result = {}
        for i, agent in enumerate(self.agents):
            # Add one-hot agent ID so shared policy can distinguish roles
            agent_id = np.zeros(len(self.agents), dtype=np.int32)
            agent_id[i] = 1.0
            result[agent] = np.concatenate([obs_array, agent_id])
        return result

    def close(self):
        self.client.Close(grid_world_pb2.GWCloseRequest(id=self.id))

    def reset(self, seed=None, options=None):
        state = None
        self.pursuer = []
        self.evader = None
        self.timestep = 0

        if self.id is None:
            response = self.client.New(grid_world_pb2.GWNewRequest())
            self.id = response.id
            state = response.state
        else:
            response = self.client.Reset(grid_world_pb2.GWResetRequest(id=self.id))
            state = response.state

        for drone_state in state.drone_states:
            if drone_state.is_evader:
                self.evader = Drone(drone_state.id, drone_state.x, drone_state.y, True)
            else:
                self.pursuer.append(
                    Drone(drone_state.id, drone_state.x, drone_state.y, False)
                )

        if self.evader is None or len(self.pursuer) == 0:
            raise ValueError("Pursuer or evader not initialized after reset")
        self.agents = [drone.name for drone in self.pursuer] + [self.evader.name]

        observations = self._get_obs()

        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions: dict[str, float]):
        if self.evader is None or len(self.pursuer) == 0:
            raise ValueError("Pursuer or evader not initialized")

        if self.step_delay > 0:
            time.sleep(self.step_delay)
        rewards: dict[str, float] = {a: 0 for a in self.agents}
        terminations: dict[str, bool] = {a: False for a in self.agents}
        truncations: dict[str, bool] = {a: False for a in self.agents}

        # Create the list of actions to send to the godot server
        actions_send = [
            grid_world_pb2.GWDroneAction(
                id=self.evader.id, action=get_action(actions["evader"])
            ),
        ]
        for i, pursuer in enumerate(
            self.pursuer
        ):  # add the pursuer actions if they are not destroyed
            if not pursuer.destroyed:
                actions_send.append(
                    grid_world_pb2.GWDroneAction(
                        id=pursuer.id, action=get_action(actions[f"pursuer_{i}"])
                    )
                )

        response = self.client.DoStep(
            grid_world_pb2.GWActionRequest(id=self.id, drone_actions=actions_send)
        )

        if response.WhichOneof("state_or_error") == "state":
            state = response.state
            for drone_state in state.drone_states:
                if (
                    drone_state.destroyed and not drone_state.is_evader
                ):  # Pursuer destroyed
                    for pursuer in self.pursuer:
                        if pursuer.id == drone_state.id:
                            pursuer.destroyed = True
                            rewards[pursuer.name] += REWARD_PURSUER_DESTROYED
                if drone_state.is_evader:  # Update evader position
                    self.evader.x = drone_state.x
                    self.evader.y = drone_state.y
                else:
                    for pursuer in self.pursuer:  # Update pursuer position
                        if pursuer.id == drone_state.id:
                            pursuer.x = drone_state.x
                            pursuer.y = drone_state.y
        else:
            raise ValueError("Error in step")

        if response.state.terminated:
            if (
                self.evader.x == self.target_x and self.evader.y == self.target_y
            ):  # Evader reached target
                rewards[self.evader.name] += REWARD_EVADER_TARGET_REACHED
                for pursuer in self.pursuer:
                    rewards[pursuer.name] += REWARD_PURSUER_TARGET_REACHED
                terminations = {a: True for a in self.agents}
            elif any(
                pursuer.x == self.evader.x and pursuer.y == self.evader.y
                for pursuer in self.pursuer
            ):  # Evader caught by pursuer
                rewards[self.evader.name] += REWARD_EVADER_CAUGHT
                # reward only the pursuer that caught the evader
                for pursuer in self.pursuer:
                    if pursuer.x == self.evader.x and pursuer.y == self.evader.y:
                        rewards[pursuer.name] += REWARD_PURSUER_CAUGHT_EVADER_SELF
                    else:
                        rewards[pursuer.name] += REWARD_PURSUER_CAUGHT_EVADER_OTHERS
                terminations = {a: True for a in self.agents}
            elif (
                self.evader.x > 10
                or self.evader.y > 10
                or self.evader.x < 0
                or self.evader.y < 0
            ):  # Evader out of bounds
                rewards[self.evader.name] += REWARD_EVADER_OUT_OF_BOUNDS
                terminations = {a: True for a in self.agents}

        if self.timestep >= 100:  # Max timesteps reached
            rewards[self.evader.name] += REWARD_EVADER_MAX_TIMESTEPS
            for pursuer in self.pursuer:
                rewards[pursuer.name] += REWARD_PURSUER_MAX_TIMESTEPS
            truncations = {a: True for a in self.agents}

        # punish the pursuers for being far from the evader to encourage them to move towards the evader
        for pursuer in self.pursuer:
            distance_to_evader = np.sqrt(
                (pursuer.x - self.evader.x) ** 2 + (pursuer.y - self.evader.y) ** 2
            )
            rewards[pursuer.name] += distance_to_evader * REWARD_PURSUER_FAR_FROM_EVADER

        # punish the evader for being far from the target to encourage it to move towards the target
        distance_to_target = np.sqrt(
            (self.evader.x - self.target_x) ** 2 + (self.evader.y - self.target_y) ** 2
        )
        rewards[self.evader.name] += distance_to_target * REWARD_EVADER_FAR_FROM_TARGET

        # reward the evader for being far from the pursuers to encourage it to move away from the pursuers
        closest_pursuer_distance = min(
            np.sqrt((pursuer.x - self.evader.x) ** 2 + (pursuer.y - self.evader.y) ** 2)
            for pursuer in self.pursuer
        )
        rewards[self.evader.name] += (
            closest_pursuer_distance * REWARD_EVADER_FAR_FROM_PUSUERS
        )

        self.timestep += 1

        observations = self._get_obs()

        infos = {a: {} for a in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    def observation_space(self, agent):
        n_pursuers = len(self.agents) - 1
        n_positions = (
            n_pursuers + 2
        ) * 2  # x,y pairs for each pursuer + evader + target
        n_agent_id = len(
            self.agents
        )  # one-hot agent ID, so drones can share a policy but still know who they are
        return Box(
            low=0.0, high=1.0, shape=(n_positions + n_agent_id,), dtype=np.float32
        )

    def action_space(self, agent):
        return Discrete(5)  # 5 possible actions: up, down, left, right, stay
