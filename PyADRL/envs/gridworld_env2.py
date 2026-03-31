from pettingzoo import ParallelEnv
import grpc
from .. import grid_world_pb2
from ..utils.protobuf_utils import get_action
from ..utils.gridworld_client import GridWorldClient
from ..logger.metricslogger import EpisodeOutcome
from gymnasium.spaces import Box, Discrete
import numpy as np
import time
from .reward_functions.rewards import RewardFunction

class Drone:
    def __init__(self, id: int, x: int, y: int, is_evader: bool):
        self.id: int = id
        self.x = x
        self.y = y
        self.name = f"evader_{id}" if is_evader else f"pursuer_{id}"
        self.is_evader = is_evader
        self.destroyed: bool = False

class GridWorldEnvironment(ParallelEnv):
    metadata = {
        "name": "gridworld_environment_v1",
    }

    # map_size should be changed once we have a common way to define map attributes
    def __init__(self, channel: grpc.Channel, 
                 n_pursuers, n_evaders: int,
                 map_size: int, target_x: int, target_y: int,
                 reward_function: RewardFunction, max_time: int = 100, step_delay: float = 0.0):
        self.id: int | None = None
        self.client = GridWorldClient(channel)
        self.step_delay: float = step_delay

        self.map_size: int = map_size
        self.target_x: int = target_x
        self.target_y: int = target_y

        # does it make sense to make drones a dictionary?
        # possibly with a key on name to remove agents
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.drones = {"evaders": [], "pursuers": []}

        self.reward_function = reward_function
        self.timestep: int = 0
        self.max_time:int = max_time
        self.agents = []

    def _get_obs(self):
        if self.evader is None or len(self.pursuer) == 0:
            raise ValueError("Pursuer or evader not initialized")

        obs = []
        for p in self.pursuer:
            obs += [p.x / self.map_size, p.y / self.map_size]
        for e in self.evaders:
            obs += [e.x / self.map_size, e.y / self.map_size]
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
        self.drones = {"evaders": [], "pursuers": []}
        self.pursuers = []
        self.evaders = []
        self.timestep = 0

        if self.id is None:
            response = self.client.New(grid_world_pb2.GWNewRequest(
                map_size=self.map_size,
                target_x=self.target_x,
                target_y=self.target_y,
                evader_count=self.n_evaders,
                pursuer_count=self.n_pursuers,
            ))
            self.id = response.id
            state = response.state
        else:
            response = self.client.Reset(grid_world_pb2.GWResetRequest(id=self.id))
            state = response.state

        for drone_state in state.drone_states:
            is_evader = drone_state.is_evader
            drone = Drone(drone_state.id, drone_state.x, drone_state.y, is_evader)
            drone_name = drone.name
            self.agents.append(drone_name)
            if is_evader:
                self.drones["evaders"].append(drone)
            else:
                self.drones["pursuers"].append(drone)

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
        terminations: dict[str, bool] = {a: False for a in self.agents}
        truncations: dict[str, bool] = {a: False for a in self.agents}

        # Create the list of actions to send to the godot server
        actions_send = []
        # add the drone actions if they are not destroyed
        for i, evader in enumerate(self.evaders):
            if not evader.destroyed:
                actions_send.append(
                    grid_world_pb2.GWDroneAction(
                        id=evader.id, action=get_action(actions[f"evader_{i}"])
                    )
                )
        for i, pursuer in enumerate(self.pursuer):  
            if not pursuer.destroyed:
                actions_send.append(
                    grid_world_pb2.GWDroneAction(
                        id=pursuer.id, action=get_action(actions[f"pursuer_{i}"])
                    )
                )

        response = self.client.DoStep(
            grid_world_pb2.GWActionRequest(id=self.id, drone_actions=actions_send)
        )

        # Ugh fix
        if response.WhichOneof("state_or_error") == "state":
            state = response.state
            for drone_state in state.drone_states:
                if (
                    drone_state.destroyed and not drone_state.is_evader
                ):  # Pursuer destroyed
                    for pursuer in self.pursuer:
                        if pursuer.id == drone_state.id and not pursuer.destroyed:
                            pursuer.destroyed = True
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

        # TODO: add capture rate to the outcome
        outcome = EpisodeOutcome(episode_length=self.timestep + 1)

        # TODO: Add parameters
        rewards = self.reward_function.calculate_rewards()
        
        if self.timestep >= self.max_time:
            truncations = {a: True for a in self.agents}
        if response.state.terminated:
            terminations = {a: True for a in self.agents}

        self.timestep += 1

        infos = {a: {} for a in self.agents}

        is_done = any(terminations.values()) or any(truncations.values())
        if is_done:
            episode_metrics = {
                "captured": 1.0 if outcome.captured else 0.0,
                "breached": 1.0 if outcome.breached else 0.0,
                "capture_step": float(outcome.capture_step)
                if outcome.capture_step is not None
                else 100.0, # What is this?
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

        # x,y pairs for each pursuer + evader + target
        n_positions = (n_pursuers + 2) * 2 

        # one-hot agent ID, so drones can share a policy but still know who they are
        n_agent_id = len(self.agents) # Isn't this the same as n_pursuers + n_evaders?

        return Box(
            low=0.0, high=1.0, shape=(n_positions + n_agent_id,), dtype=np.float32
        )

    def action_space(self, agent):
        return Discrete(5) # 5 possible actions: up, down, left, right, stay
