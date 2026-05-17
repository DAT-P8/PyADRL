from pettingzoo import ParallelEnv
from gymnasium.spaces import Box, Discrete
import numpy as np
import time
import grpc
from ..utils.ngw2d_actions import get_drone_action
from PyADRL.shielding.centralized_shield import CentralizedShield
from PyADRL.shielding.shield import Shield
from ..utils.ngw2d_client import NGWClient
from .reward_functions.rewards import RewardFunction
from .map_configs.map_config import MapConfig
from .ngw_drone import NGW_Drone
from ..dtos.ngw_dtos import (
    DroneAction,
    State,
)

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
        shielded: bool = False,
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
        self.newest_state: State | None = None

        # Pettingzoo wants all agents to have the same observation space, action space,
        # and wants possible agents to be defined
        self.possible_agents = [
            f"{agent_type}_{i}"
            for agent_type in ["evader", "pursuer"]
            for i in range(n_pursuers + n_evaders)
        ]

        # one-hot agent ID, so drones can share a policy but still know who they are
        n_agents = self.n_evaders + self.n_pursuers

        self.n_objects_positions = (
            len(map_config.get_objects()) * 2
        )  # x, y for each object
        self.objects_state = []

        target_position = 2  # x and y position of the target

        # x,y pairs for each agent, the target tile, and objects
        n_agent_positions = n_agents * 2
        role_bits = n_agents  # not actually sure if role_bits does anything
        one_hot = n_agents  # one hot encoding for agent ID, so shared policy can distinguish agents
        self.obs_space = Box(
            low=0.0,
            high=1.0,
            shape=(n_agent_positions + one_hot + role_bits + target_position,),
            dtype=np.float32,
        )

        # 1 for nothing action, 8 * velocity for all actions with an actual direction
        self.act_space = Discrete(1 + (8 * drone_velocity))

        self.shield: Shield | None = None
        if shielded:
            self.shield = CentralizedShield(map=map_config.get_map_spec())

    def _get_obs(self):
        obs = []
        for p in self.drones[PURSUERS]:
            (norm_x, norm_y) = self.map_config.normalise_position(p.x, p.y)
            obs += [norm_x, norm_y]
        for e in self.drones[EVADERS]:
            (norm_x, norm_y) = self.map_config.normalise_position(e.x, e.y)
            obs += [norm_x, norm_y]

        # Include the fixed target position so the observation matches the declared space.
        obs += [self.norm_target_x, self.norm_target_y]

        # Role bits, 1 for pursuer, 0 for evader
        obs += [1.0 for _ in self.drones[PURSUERS]] + [
            0.0 for _ in self.drones[EVADERS]
        ]

        obs_array = np.array(obs, dtype=np.float32)

        agent_observations = {}
        for agent in self.agents:
            one_hot_agent = self.one_hot[agent]
            agent_observations[agent] = np.concatenate([obs_array, one_hot_agent])

        return agent_observations

    def close(self):
        if self.id is None:
            raise ValueError("Tried closing a null environment")
        self.client.Close(self.id)

    def reset(self, seed=None, options=None):
        self.drones = {EVADERS: [], PURSUERS: []}
        self.agents = []
        self.timestep = 0

        if self.id is None:
            self.newest_state = self.client.New(
                self.map_config.get_map_spec(), self.n_evaders, self.n_pursuers
            )
            self.id = self.newest_state.sim_id
        else:
            try:
                self.newest_state = self.client.Reset(self.id)
            except (ValueError, grpc.RpcError) as e:
                # Two recovery paths, same response:
                #   ValueError: server returned a structured error in the response
                #     body (e.g. "simulation doesn't exist" after a TTL reap).
                #   grpc.RpcError: server handler threw an unhandled exception
                #     (e.g. spawn-position generation hit its iteration cap due
                #     to a thread-unsafe RNG, or any other server-side handler
                #     fault). The sim may be in an inconsistent state.
                # In both cases, discard the sim and create a fresh one rather
                # than killing the whole training trial.
                print(
                    f"Reset failed for sim {self.id} ({type(e).__name__}: {e}); "
                    f"creating a new sim"
                )
                self.newest_state = self.client.New(
                    self.map_config.get_map_spec(),
                    self.n_evaders,
                    self.n_pursuers,
                )
                self.id = self.newest_state.sim_id

        for drone_state in self.newest_state.drone_states:
            is_evader = drone_state.is_evader
            drone = NGW_Drone(drone_state.id, drone_state.x, drone_state.y, is_evader)
            self.agents.append(drone.name)
            if is_evader:
                self.drones[EVADERS].append(drone)
            else:
                self.drones[PURSUERS].append(drone)

        self.objects_state = self.newest_state.objects

        if len(self.drones[EVADERS]) == 0 or len(self.drones[PURSUERS]) == 0:
            raise ValueError(
                f"Pursuer or evader not initialized after reset\n{self.drones}"
            )
        if len(self.agents) == 0:
            raise ValueError("No agents initialised")

        self.one_hot = self.encode_one_hot(self.agents)
        observations = self._get_obs()

        infos = {a: {} for a in self.agents}
        for d in self.drones[EVADERS] + self.drones[PURSUERS]:
            if d.name in infos:
                infos[d.name]["drone"] = d

        return (observations, infos)

    def step(self, actions: dict[str, float]):
        if (
            len(self.drones[EVADERS]) == 0
            or len(self.drones[PURSUERS]) == 0
            or self.id is None
        ):
            raise ValueError("Pursuer or evader not initialized")

        if self.step_delay > 0:
            time.sleep(self.step_delay)

        # Create the list of actions to send to the godot server
        actions_send: list[DroneAction] = []
        # Add drone action if it is not destroyed
        for drones in self.drones.values():
            for d in drones:
                if not d.destroyed:
                    actions_send.append(get_drone_action(actions[d.name], d))

        alt_state: State | None = None
        if self.shield is not None:
            if self.newest_state is not None:
                new_acts, alt_state = self.shield.shield(
                    actions_send, self.newest_state
                )
                actions_send = new_acts
            else:
                raise Exception("Tried shielding on a stateless simulation?")

        self.newest_state = self.client.DoStep(self.id, actions_send)
        name_to_drone = {k.name: k for drones in self.drones.values() for k in drones}

        terminations: dict[str, bool] = {a: False for a in self.agents}
        truncations: dict[str, bool] = {a: False for a in self.agents}
        terminations["__all__"] = False
        truncations["__all__"] = False

        for drone_state in self.newest_state.drone_states:
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

        time_limit_reached = self.timestep >= self.time_limit
        rewards = self.reward_function.calculate_rewards(
            events=alt_state.events
            if alt_state is not None
            else self.newest_state.events,
            drones=[drone for drone in self.newest_state.drone_states],
            map_config=self.map_config,
            time_limit_reached=time_limit_reached,
        )

        if self.newest_state.terminated:
            terminations = {a: True for a in self.agents}
            terminations["__all__"] = True
        elif time_limit_reached:
            truncations = {a: True for a in self.agents}
            truncations["__all__"] = True

        self.timestep += 1

        infos = {a: {} for a in self.agents}

        observations = self._get_obs()

        for d in self.drones[EVADERS] + self.drones[PURSUERS]:
            if d.name in infos:
                infos[d.name]["drone_state"] = {
                    "x": d.x,
                    "y": d.y,
                    "destroyed": d.destroyed,
                }
                infos[d.name]["events"] = self.newest_state.events
                if alt_state is not None:
                    infos[d.name]["shield_events"] = alt_state.events

        name_to_reward: dict[str, float] = {
            name: rewards[name_to_drone[name].id] for name in self.agents
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

        return observations, name_to_reward, terminations, truncations, infos

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
