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
from ..utils.ngw2d_client import (
    NGWClient,
    State,
)
from .reward_functions.rewards import RewardFunction
from .map_configs.map_config import MapConfig
from ..logger.metricslogger import EpisodeOutcome
from .ngw_drone import NGW_Drone

# Drone dictionary names
EVADERS = "evaders"
PURSUERS = "pursuers"


class NGWEnvironment(ParallelEnv):
    metadata = {
        "name": "new_gridworld_environment_v1",
    }

    def __init__(
        self,
        channel: grpc.Channel,
        map_config: MapConfig,
        reward_function: RewardFunction,
        n_pursuers: int,
        n_evaders: int,
        drone_velocity: int = 1,
        time_limit: int = 100,
        step_delay: float = 0.0,
    ):
        self.id: int | None = None
        self.client = NGWClient(channel)
        self.step_delay = step_delay

        self.map_config = map_config
        # Calculate normalised target position here to avoid doing it each _get_obs call
        (target_x, target_y) = map_config.get_target_position()
        self.norm_target_x, self.norm_target_y = map_config.normalise_position(
            target_x, target_y
        )

        self.drones: dict[str, list[NGW_Drone]] = {EVADERS: [], PURSUERS: []}
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.drone_velocity = drone_velocity

        self.reward_function = reward_function
        self.timestep = 0
        self.time_limit = time_limit
        self.agents = []
        self.one_hot = {}

        # Pettingzoo wants all agents to have the same observation space, action space,
        # and wants possible agents to be defined
        self.possible_agents = [
            f"{agent_type}_{i}"
            for agent_type in ["evader", "pursuer"]
            for i in range(n_pursuers + n_evaders)
        ]

        # one-hot agent ID, so drones can share a policy but still know who they are
        n_agents = self.n_evaders + self.n_pursuers
        # x,y pairs for each agent and the target tile
        n_positions = (n_agents + 1) * 2
        self.obs_space = Box(
            low=0.0, high=1.0, shape=(n_positions + n_agents,), dtype=np.float32
        )

        # 9 possible actions
        self.act_space = Discrete(9)

    def _get_obs(self):
        obs = []
        for p in self.drones[PURSUERS]:
            (norm_x, norm_y) = self.map_config.normalise_position(p.x, p.y)
            obs += [norm_x, norm_y]
        for e in self.drones[EVADERS]:
            (norm_x, norm_y) = self.map_config.normalise_position(e.x, e.y)
            obs += [norm_x, norm_y]
        obs += [self.norm_target_x, self.norm_target_y]
        obs_array = np.array(obs, dtype=np.float32)

        agent_observations = {}
        for agent in self.agents:
            one_hot_agent = self.one_hot[agent]
            agent_observations[agent] = np.concatenate([obs_array, one_hot_agent])

        return agent_observations

    def close(self):
        self.client.Close(CloseRequest(sim_id=self.id))

    def reset(self, seed=None, options=None):
        state = None
        self.drones = {EVADERS: [], PURSUERS: []}
        self.agents = []
        self.timestep = 0

        if self.id is None:
            state = self.client.New(
                NewRequest(
                    map=self.map_config.get_map_spec(),
                    evader_count=self.n_evaders,
                    pursuer_count=self.n_pursuers,
                    drone_velocity=self.drone_velocity,
                )
            )
            self.id = state.sim_id
        else:
            state = self.client.Reset(ResetRequest(sim_id=self.id))

        for drone_state in state.drone_states:
            is_evader = drone_state.is_evader
            drone = NGW_Drone(drone_state.id, drone_state.x, drone_state.y, is_evader)
            self.agents.append(drone.name)
            if is_evader:
                self.drones[EVADERS].append(drone)
            else:
                self.drones[PURSUERS].append(drone)

        if len(self.drones[EVADERS]) == 0 or len(self.drones[PURSUERS]) == 0:
            raise ValueError(
                f"Pursuer or evader not initialized after reset\n{self.drones}"
            )
        if len(self.agents) == 0:
            raise ValueError("No agents initialised")

        self.one_hot = self.encode_one_hot(self.agents)
        observations = self._get_obs()

        infos = {a: {} for a in self.agents}

        return (observations, infos)

    def step(self, actions: dict[str, float]):
        if len(self.drones[EVADERS]) == 0 or len(self.drones[PURSUERS]) == 0:
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
                            id=d.id,
                            action=get_action(actions[d.name]),
                            velocity=self.drone_velocity,
                        )
                    )

        state = self.client.DoStep(
            DoStepRequest(sim_id=self.id, drone_actions=actions_send)
        )

        terminations: dict[str, bool] = {a: False for a in self.agents}
        truncations: dict[str, bool] = {a: False for a in self.agents}
        terminations["__all__"] = False
        truncations["__all__"] = False

        for drone_state in state.drone_states:
            key = EVADERS if drone_state.is_evader else PURSUERS
            # Mark destroyed drones as destroyed in python
            if drone_state.destroyed:
                for drone in self.drones[key]:
                    if drone_state.id == drone.id:
                        drone.destroyed = True
                        terminations[drone.name] = True
                        break
            else:  # Update positions
                for drone in self.drones[key]:
                    if drone_state.id == drone.id:
                        drone.x = drone_state.x
                        drone.y = drone_state.y

        outcome = EpisodeOutcome(episode_length=self.timestep + 1)

        time_limit_reached = self.timestep >= self.time_limit
        rewards = self.reward_function.calculate_rewards(
            new_state=state,
            agents=self.agents,
            drones=self.drones,
            map_config=self.map_config,
            time_limit_reached=time_limit_reached,
        )

        if state.terminated:
            terminations = {a: True for a in self.agents}
            terminations["__all__"] = True
        elif time_limit_reached:
            truncations = {a: True for a in self.agents}
            truncations["__all__"] = True

        self.timestep += 1

        infos = {a: {} for a in self.agents}

        if state.terminated or time_limit_reached:
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

        for d in self.drones[EVADERS] + self.drones[PURSUERS]:
            if d.name in infos:
                infos[d.name]["drone_state"] = {
                    "x": d.x,
                    "y": d.y,
                    "destroyed": d.destroyed,
                }

        # Remove terminated agents
        if truncations["__all__"] or terminations["__all__"]:
            self.agents = []
        else:
            self.agents = [
                d.name
                for d in (self.drones[EVADERS] + self.drones[PURSUERS])
                if not d.destroyed
            ]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass

    def observation_space(self, agent):
        return self.obs_space

    def action_space(self, agent):
        return self.act_space

    def encode_one_hot(self, agents):
        one_hot = {}
        for i, agent in enumerate(agents):
            # Add one-hot agent ID so shared policy can distinguish roles
            agent_id = np.zeros(len(agents), dtype=np.int32)
            agent_id[i] = 1.0
            one_hot[agent] = agent_id
        return one_hot
