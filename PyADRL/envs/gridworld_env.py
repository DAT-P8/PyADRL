"""Gridworld MA area defense environment"""

from pettingzoo import ParallelEnv
import grpc
from .. import grid_world_pb2
from ..utils.protobuf_utils import get_action
from ..utils.gridworld_client import GridWorldClient
from ..logger.metricslogger import EpisodeOutcome
from gymnasium.spaces import Box, Discrete
import numpy as np
import time


class Drone:
    def __init__(self, id: int, x: int, y: int, is_evader: bool):
        self.id: int = id
        self.x = x
        self.y = y
        self.is_evader = is_evader
        self.destroyed: bool = False


class GridWorldEnvironment(ParallelEnv):
    metadata = {
        "name": "gridworld_environment_v0",
    }

    def __init__(
        self,
        channel: grpc.Channel,
        step_delay: float = 0.0,
        max_episode_length: int = 100,
    ):
        self.id: int | None = None
        self.client = GridWorldClient(channel)
        self.step_delay = step_delay  # Delay to slow down steps for visualization
        self.target_x: int = 5
        self.target_y: int = 5
        self.pursuer: list[Drone] = []
        self.evader: Drone | None = None
        self.timestep: int = 0
        self.max_episode_length: int = max_episode_length
        self.possible_agents = ["pursuer_0", "pursuer_1", "evader"]
        self.agents = self.possible_agents.copy()

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
        for i, agent in enumerate(self.possible_agents):
            # Add one-hot agent ID so shared policy can distinguish roles
            agent_id = np.zeros(len(self.possible_agents), dtype=np.int32)
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
        self.agents = self.possible_agents.copy()

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

        observations = self._get_obs()

        infos = {a: {} for a in self.possible_agents}

        return observations, infos

    def step(self, actions: dict[str, float]):
        if self.evader is None or len(self.pursuer) == 0:
            raise ValueError("Pursuer or evader not initialized")

        if self.step_delay > 0:
            time.sleep(self.step_delay)
        pursuer_reward: float = 0
        evader_reward: float = 0
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}

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
                    pursuer_reward -= 100
                if drone_state.is_evader:  # Update evader state
                    self.evader.x = drone_state.x
                    self.evader.y = drone_state.y
                    self.evader.destroyed = drone_state.destroyed
                else:
                    for pursuer in self.pursuer:  # Update pursuer position
                        if pursuer.id == drone_state.id:
                            pursuer.x = drone_state.x
                            pursuer.y = drone_state.y
        else:
            raise ValueError("Error in step")

        outcome = EpisodeOutcome(episode_length=self.timestep + 1)

        if response.state.terminated:
            out_of_bounds = (
                self.evader.x > 10
                or self.evader.y > 10
                or self.evader.x < 0
                or self.evader.y < 0
            )
            breached = self.evader.x == self.target_x and self.evader.y == self.target_y
            captured = self.evader.destroyed and not out_of_bounds
            if out_of_bounds:
                evader_reward -= 1000
                terminations = {a: True for a in self.possible_agents}
            elif captured:
                evader_reward -= 10
                pursuer_reward += 1000
                terminations = {a: True for a in self.possible_agents}
                outcome.captured = True
                outcome.capture_step = self.timestep + 1
            elif breached:
                evader_reward += 1000
                pursuer_reward -= 100
                terminations = {a: True for a in self.possible_agents}
                outcome.breached = True
            elif out_of_bounds:
                evader_reward -= 1000
                terminations = {a: True for a in self.possible_agents}
            else:
                terminations = {a: True for a in self.possible_agents}

        if self.timestep >= self.max_episode_length:  # Max timesteps reached
            truncations = {a: True for a in self.possible_agents}

        pursuer_reward -= 1
        evader_reward -= 1

        self.timestep += 1

        infos = {a: {} for a in self.possible_agents}

        is_done = any(terminations.values()) or any(truncations.values())
        if is_done:
            episode_metrics = {
                "captured": 1.0 if outcome.captured else 0.0,
                "breached": 1.0 if outcome.breached else 0.0,
                "capture_step": float(outcome.capture_step)
                if outcome.capture_step is not None
                else float(self.max_episode_length),
                "episode_length": float(outcome.episode_length),
            }
            for a in self.possible_agents:
                infos[a]["episode_metrics"] = episode_metrics

        observations = self._get_obs()

        rewards: dict[str, float] = {a: 0 for a in self.possible_agents}
        rewards["pursuer_0"] = pursuer_reward
        rewards["pursuer_1"] = pursuer_reward
        rewards["evader"] = evader_reward

        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    def observation_space(self, agent):
        n_pursuers = len(self.possible_agents) - 1
        n_positions = (
            n_pursuers + 2
        ) * 2  # x,y pairs for each pursuer + evader + target
        n_agent_id = len(
            self.possible_agents
        )  # one-hot agent ID, so drones can share a policy but still know who they are
        return Box(
            low=0.0, high=1.0, shape=(n_positions + n_agent_id,), dtype=np.float32
        )

    def action_space(self, agent):
        return Discrete(5)  # 5 possible actions: up, down, left, right, stay
