"""Gridworld MA area defense environment"""

from pettingzoo import ParallelEnv
import grpc
from .. import grid_world_pb2, SimStates
from ..utils.protobuf_utils import get_action
from ..utils.gridworld_client import GridWorldClient
from ..logger.metricslogger import EpisodeOutcome
from gymnasium.spaces import Box, Discrete
import numpy as np
import time
from reward_functions.rewards import GridWorldRewards

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
        "name": "gridworld_environment_v1",
    }

    def __init__(self, channel: grpc.Channel, map_size: int, target_x: int, target_y : int, max_timestep: int, step_delay: float = 0.0):
        self.id: int | None = None
        self.client = GridWorldClient(channel)
        self.reward_function = GridWorldRewards()
        self.step_delay = step_delay  # Delay to slow down steps for visualization
        
        self.map_size = map_size
        self.target_x: int = target_x
        self.target_y: int = target_y
        if self.map_size < self.target_x or self.map_size < self.target_y:
            raise ValueError("Target cannot be placed outside the map")
        self.time_limit = max_timestep

        self.pursuer: list[Drone] = []
        self.evader: Drone | None = None
        self.timestep: int = 0
        self.agents = []

    def _get_obs(self):
        if self.evader is None or len(self.pursuer) == 0:
            raise ValueError("Pursuer or evader not initialized")

        # Making an assumption that we divide by map size to normalise
        obs = []
        for p in self.pursuer:
            obs += [p.x / self.map_size, p.y / self.map_size]
        obs += [self.evader.x / self.map_size, self.evader.y / self.map_size]
        obs += [self.target_x / self.map_size, self.target_y / self.map_size]
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
            response = self.client.New(grid_world_pb2.GWNewRequest(
                map_size = self.map_size, 
                target_x = self.target_x, 
                target_y = self.target_y, 
                time_limit = self.time_limit))

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

        if response.WhichOneof("state_or_error") != "state":
            raise ValueError("Error in step")
    
        newly_destroyed = []
        for drone in response.state.drone_states:
            for pursuer in self.pursuer:
                is_newly_destroyed = (
                    drone.id == pursuer.id and
                    drone.is_destroyed and 
                    not pursuer.destroyed
                )
                if is_newly_destroyed:
                    pursuer.destroyed = True
                    newly_destroyed.append(pursuer.name)

            if drone.is_evader:  # Update evader position
                self.evader.x = drone.x
                self.evader.y = drone.y
            else:
                for pursuer in self.pursuer:  # Update pursuer position
                    if pursuer.id == drone.id:
                        pursuer.x = drone.x
                        pursuer.y = drone.y

        outcome = EpisodeOutcome(episode_length=self.timestep + 1)
        rewards = self.reward_function.calculate_rewards(
            response = response, 
            rewards = {a: 0 for a in self.agents}, 
            pursuers = self.pursuer,
            evader = self.evader,
            target_x = self.target_x,
            target_y = self.target_y,
            newly_destroyed = newly_destroyed,
        )
        self.timestep += 1

        infos = {a: {} for a in self.agents}
        terminations: dict[str, bool] = {a: False for a in self.agents}
        truncations: dict[str, bool] = {a: False for a in self.agents}
        simulation_state = response.state.simulation_state
        if simulation_state != SimStates.Running:
            time_exceeded = False
            match simulation_state:
                case SimStates.TimeExceeded:
                    truncations = {a: True for a in self.agents}
                    time_exceeded = True
                case SimStates.EvadersCaptured:
                    outcome.captured = True
                    outcome.capture_step = self.timestep # + 1
                case SimStates.TargetReached:
                    outcome.breached = True
            
            if not time_exceeded:
                terminations = {a: True for a in self.agents}

            episode_metrics = {
                "captured": 1.0 if outcome.captured else 0.0,
                "breached": 1.0 if outcome.breached else 0.0,
                "capture_step": float(outcome.capture_step)
                if outcome.capture_step is not None
                else 100.0,
                "episode_length": float(outcome.episode_length),
            }
            for a in self.agents:
                infos[a]["episode_metrics"] = episode_metrics

        observations = self._get_obs()

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
