from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete
import numpy as np
import time
import grpc
from ..ngw.v1.ngw2d_pb2 import (
    CloseRequest,
    NewRequest,
    ResetRequest,
    DroneAction,
    DoStepRequest,
)
from ..utils.ngw2d_actions import get_action
from ..utils.gridworld_client import GridWorldClient
from .reward_functions.rewards import RewardFunction
from .map_configs.map_config import MapConfig
from ..logger.metricslogger import EpisodeOutcome
from .ngw_drone import NGW_Drone

class NGWEnvironment(ParallelEnv):
    metadata = {
        "name": "new_gridworld_environment_v1",
    }

    def __init__(self, channel: grpc.Channel,
                 map_config: MapConfig,
                 reward_function: RewardFunction,
                 n_pursuers: int,
                 n_evaders: int,
                 drone_velocity: int = 1,
                 time_limit: int = 100,
                 step_delay: float = 0.0):
        self.id: int | None = None
        self.client = GridWorldClient(channel)
        self.step_delay = step_delay

        self.map_config = map_config
        (target_x,target_y) = map_config.get_target_position()
        self.norm_target_x, self.norm_target_y = map_config.normalise_position(target_x, target_y)
        #self.map_spec = SquareMapHelper(11, 11, 6, 6)

        self.drones: dict[str,NGW_Drone] = {"evaders": [], "pursuers": []}
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.drone_velocity = drone_velocity

        self.reward_function = reward_function
        self.timestep = 0
        self.time_limit = time_limit
        self.agents = []

    def _get_obs(self):
        #if len(self.drones['evaders']) == 0 or len(self.drones['pursuers']) == 0:
        #    raise ValueError("Pursuer or evader not initialized")

        obs = []
        for p in self.drones['pursuer']:
            (norm_x,norm_y) = self.map_config.normalise_position(p.x,p.y)
            obs += [norm_x, norm_y]
        for e in self.drones['evaders']:
            (norm_x,norm_y) = self.map_config.normalise_position(e.x,e.y)
            obs += [norm_x, norm_y]
            #obs += [e.x / self.map_size, e.y / self.map_size]
        obs += [self.norm_target_x, self.norm_target_y]
        #obs += [self.target_x / self.map_size, self.target_y / self.map_size]
        obs_array = np.array(obs, dtype=np.float32)

        result = {}
        for i, agent in enumerate(self.agents):
            # Add one-hot agent ID so shared policy can distinguish roles
            agent_id = np.zeros(len(self.agents), dtype=np.int32)
            agent_id[i] = 1.0
            result[agent] = np.concatenate([obs_array, agent_id])
        return result

    def close(self):
        self.client.Close(CloseRequest(id=self.id))

    def reset(self, seed=None, options=None):
        state = None
        self.drones = {'evaders': [], 'pursuers': []}
        #self.pursuers = []
        #self.evaders = []
        self.timestep = 0

        if self.id is None:
            response = self.client.New(NewRequest(
                map=self.map_config.get_map_spec(),
                evader_count=self.n_evaders,
                pursuer_count=self.n_pursuers,
                drone_velocity=self.drone_velocity,
            ))
            self.id = response.id
            state = response.state
        else:
            response = self.client.Reset(ResetRequest(id=self.id))
            state = response.state

        for drone_state in state.drone_states:
            is_evader = drone_state.is_evader
            drone = NGW_Drone(drone_state.id, drone_state.x, drone_state.y, is_evader)
            self.agents.append(drone.name)
            if is_evader:
                self.drones['evaders'].append(drone)
            else:
                self.drones['pursuers'].append(drone)

        if not len(self.drones['evaders']) == 0 or len(self.drones['pursuers']) == 0:
            raise ValueError("Pursuer or evader not initialized after reset")

        observations = self._get_obs()

        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions: dict[str, float]):
        if not len(self.drones['evaders']) == 0 or len(self.drones['pursuers']) == 0:
            raise ValueError("Pursuer or evader not initialized")

        if self.step_delay > 0:
            time.sleep(self.step_delay)

        # Create the list of actions to send to the godot server
        actions_send = []
        # Add drone action if it is not destroyed
        for drones in self.drones.values():
            for d in drones:
                if not d.destroyed:
                    actions_send.append(
                        DroneAction(
                            id=d.id, action=get_action(actions[d.name])
                        )
                    )

        response = self.client.DoStep(
            DoStepRequest (id=self.id, drone_actions=actions_send)
        )

        if response.WhichOneof("state_or_error") == "state":
            state = response.state
            for drone_state in state.drone_states:
                key = "evaders" if drone_state.is_evader else "pursuers"
                # Mark destroyed drones as destroyed in python
                if drone_state.destroyed:
                    for drone in self.drones[key]:
                        if drone_state.id == d.id:
                            drone.destroyed = True
                            break
                else: # Update positions 
                    for drone in self.drones[key]:
                        if drone_state.id == d.id:
                            drone.x = drone_state.x
                            drone.y = drone_state.y
        else:
            raise ValueError("Error in step")

        # TODO: add capture rate to the outcome
        outcome = EpisodeOutcome(episode_length=self.timestep + 1)

        # TODO: Add parameters
        time_limit_reached = self.timestep >= self.time_limit
        rewards = self.reward_function.calculate_rewards(
            new_state = response.state,
            agents=self.agents,
            drones=self.drones,
            map_config=self.map_config,
            time_limit_reached=time_limit_reached,
        )
        
        terminations: dict[str, bool] = {a: False for a in self.agents}
        truncations: dict[str, bool] = {a: False for a in self.agents}
        if time_limit_reached:
            truncations = {a: True for a in self.agents}
        if response.state.terminated:
            terminations = {a: True for a in self.agents}

        self.timestep += 1

        infos = {a: {} for a in self.agents}

        if state.terminated:
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
        # one-hot agent ID, so drones can share a policy but still know who they are
        n_agents = len(self.agents)

        # x,y pairs for each agent and the target tile
        #TODO: un-hardcode the target when it is possible to make different target shapes
        n_positions = (n_agents + 1) * 2 

        return Box(
            low=0.0, high=1.0, shape=(n_positions + n_agents,), dtype=np.float32
        )

    def action_space(self, agent):
        return Discrete(5) # 5 possible actions: up, down, left, right, stay
