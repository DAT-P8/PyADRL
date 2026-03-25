"""Continuous MA area defense environment"""

from contextlib import contextmanager
import logging
from pettingzoo import ParallelEnv
import grpc
from ..utils.tdclient import (TDClient, TDDroneAction)
from ..logger.metricslogger import EpisodeOutcome
from ..TDF_pb2 import (
    TDFState
)
from gymnasium.spaces import Box
import numpy as np
import time

EVADER_COUNT = 5
PURSUER_COUNT = 5
EVADER_DOME_RADIUS = 20
PURSUER_DOME_RADIUS = 5
ARENA_DOME_RADIUS = 25
DRONE_MAX_SPEED = 10
SEED = 1 # This should be random

MAX_TIMESTEP = 100

# Rewards for any
REWARD_DRONE_DESTROYED = -100

# Rewards for evader
REWARD_EVADER_TARGET_REACHED = 100
REWARD_EVADER_CAUGHT = -100
REWARD_EVADER_OUT_OF_BOUNDS = -1000
REWARD_EVADER_MAX_TIMESTEPS = 50
REWARD_EVADER_FAR_FROM_TARGET = -1  # Muiltiplier for distance to target
REWARD_EVADER_FAR_FROM_PUSUERS = 1  # Multiplier for distance to closest pursuer

REWARD_EVADER_DESTROYED = -1000
REWARD_PURSUER_OUT_OF_BOUNDS = -1000

# Rewards for pursuers
REWARD_PURSUER_MAX_TIMESTEPS = -100  # Punish pursuers for not catching evader in time
REWARD_PURSUER_TARGET_REACHED = -100  # Punish pursuers for letting evader reach target
REWARD_PURSUER_CAUGHT_EVADER_SELF = 100  # Reward for catching the evader yourself
REWARD_PURSUER_CAUGHT_EVADER_OTHERS = 10  # Reward for helping catch the evader
REWARD_PURSUER_FAR_FROM_EVADER = -1  # Multiplier for distance to evader

REWARD_PURSUER_DESTROYED = -1000
REWARD_PURSUER_OUT_OF_BOUNDS = -1000


class Drone:
    def __init__(self, id: int, x: float, y: float, z: float, xv: float, yv: float, zv: float, is_evader: bool):
        self.id: int = id

        self.x = x
        self.y = y
        self.z = z

        self.xv = xv
        self.yv = yv
        self.zv = zv

        self.name = f"evader_{id}" if is_evader else f"pursuer_{id}"
        self.is_evader = is_evader
        self.destroyed: bool = False


class ThreeDEnvironment(ParallelEnv):
    metadata = {
        "name": "3d_environment_v0",
    }

    def __init__(self, channel: grpc.Channel, step_delay: float = 0.0):
        self.id: int | None = None
        self.client = TDClient(channel)
        self.step_delay = step_delay  # Delay to slow down steps for visualization
        self.target_x: int = 5
        self.target_y: int = 5
        self.pursuers: list[Drone] = []
        self.evaders: list[Drone] = []
        self.timestep: int = 0
        self.agents = []

    def _get_obs(self):
        if len(self.evaders) == 0 or len(self.pursuers) == 0:
            raise ValueError("Pursuers or evaders not initialized")

        obs: list[float] = []
        for p in self.pursuers + self.evaders:
            obs += [p.x, p.y, p.z, p.xv, p.yv, p.zv]

        obs_array = np.array(obs, dtype=np.float32)

        result = {}
        for i, agent in enumerate(self.agents):
            # Add one-hot agent ID so shared policy can distinguish roles
            agent_id = np.zeros(len(self.agents), dtype=np.int32)
            agent_id[i] = 1.0
            result[agent] = np.concatenate([obs_array, agent_id])

        return result

    def close(self):
        if self.id == None:
            logging.warning("Close called but id has not been set")
            return

        self.client.Close(self.id)

    def _update_state(self, new_state: TDFState):
        all_drones = {d.id: d for d in self.pursuers + self.evaders}
        for d_state in new_state.drone_states:
            drone = all_drones[d_state.id]
            if drone is None:
                raise Exception(f"Drone with id {id} not found")

            drone.destroyed = d_state.destroyed
            drone.x = d_state.x
            drone.y = d_state.y
            drone.z = d_state.z
            drone.xv = d_state.x_v
            drone.yv = d_state.y_v
            drone.zv = d_state.z_v

    def _reward(self, new_state: TDFState) -> dict[str, float]:
        rewards: dict[str, float] = {a: 0 for a in self.agents}

        all_drones = {d.id: d for d in self.evaders + self.pursuers}
        
        # punish the pursuers for being far from the evader to encourage them to move towards the evader
        # TODO: above

        # punish the evader for being far from the target to encourage it to move towards the target
        # TODO: above

        # reward the evader for being far from the pursuers to encourage it to move away from the pursuers
        # TODO: above

        for event in new_state.events:
            which_one = event.WhichOneof("event_case")

            if which_one == "target_reached":
                id = event.target_reached.drone_id
                d = all_drones[id]
                if d is None:
                    raise Exception(f"Drone with id {id} not found")
                rewards[d.name] += REWARD_EVADER_TARGET_REACHED

            elif which_one == "drone_out_of_bounds":
                id = event.drone_out_of_bounds.drone_id
                d = all_drones[id]
                if d is None:
                    raise Exception(f"Drone with id {id} not found")
                if d.is_evader:
                    rewards[d.name] += REWARD_EVADER_OUT_OF_BOUNDS
                else:
                    rewards[d.name] += REWARD_PURSUER_OUT_OF_BOUNDS

            elif which_one == "collision":
                ids = [d for d in event.collision.drone_ids]
                for id in ids:
                    d = all_drones[id]
                    if d is None:
                        raise Exception(f"Drone with id {id} not found")
                    if d.is_evader:
                        rewards[d.name] += REWARD_EVADER_DESTROYED
                    else:
                        rewards[d.name] += REWARD_PURSUER_DESTROYED

        if self.timestep >= MAX_TIMESTEP:
            for e in self.evaders:
                rewards[e.name] += REWARD_EVADER_MAX_TIMESTEPS

            for pursuer in self.pursuers:
                rewards[pursuer.name] += REWARD_PURSUER_MAX_TIMESTEPS

        return rewards

    def reset(self, seed=None, options=None):
        state = None
        self.pursuers = []
        self.evaders = []
        self.timestep = 0

        if self.id is None:
            responseResult = self.client.New(
                evader_count=EVADER_COUNT,
                pursuer_count=PURSUER_COUNT,
                evader_dome_radius=EVADER_DOME_RADIUS,
                pursuer_dome_radius=PURSUER_DOME_RADIUS,
                arena_dome_radius=ARENA_DOME_RADIUS,
                drone_max_speed=DRONE_MAX_SPEED,
                seed=SEED
            )
            if not responseResult.is_ok():
                raise Exception(f"Called new but got an error: {responseResult.err()}")

            response = responseResult.ok()
            self.id = response.sim_id
            state = response.drone_states
        else:
            responseResult = self.client.Reset(id=self.id)
            if not responseResult.is_ok():
                raise Exception(f"Called Reset but got an error: {responseResult.err()}")
            response = responseResult.ok()
            state = response.drone_states

        for drone_state in state:
            drone = Drone(
                drone_state.id,
                drone_state.x,
                drone_state.y,
                drone_state.z,
                drone_state.x_v,
                drone_state.y_v,
                drone_state.z_v,
                drone_state.is_evader
            )

            if drone.is_evader:
                self.evaders.append(drone)
            else:
                self.pursuers.append(drone)

        if len(self.evaders) == 0 or len(self.pursuers) == 0:
            raise ValueError("Pursuer or evader not initialized after reset")

        self.agents = [drone.name for drone in self.pursuers] + [drone.name for drone in self.evaders]

        observations = self._get_obs()

        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        print(f"ACTIONS: {"\n\t".join([f"{a}: {actions[a]}" for a in actions])}")
        if len(self.evaders) == 0 or len(self.pursuers) == 0:
            raise ValueError("Pursuer or evader not initialized")

        if self.step_delay > 0:
            time.sleep(self.step_delay)

        # Create the list of actions to send to the godot server
        evader_actions = [TDDroneAction(
            id=e.id,
            x_f=actions[e.name][0], # TODO: Get action
            y_f=actions[e.name][1], # TODO: Get action
            z_f=actions[e.name][2]  # TODO: Get action
        ) for e in self.evaders]

        pursuer_actions = [TDDroneAction(
            id=e.id,
            x_f=actions[e.name][0], # TODO: Get action
            y_f=actions[e.name][1], # TODO: Get action
            z_f=actions[e.name][2]  # TODO: Get action
        ) for e in self.pursuers]

        if self.id is None:
            raise Exception("Wanted to do DoStep but id was null")

        response_result = self.client.DoStep(
            id=self.id,
            actions=evader_actions + evader_actions
        )

        if not response_result.is_ok():
            raise Exception(f"Called DoStep but got an error: {response_result.err()}")

        response = response_result.ok()

        terminations: dict[str, bool] = {a: False for a in self.agents}
        truncations: dict[str, bool] = {a: False for a in self.agents}
        rewards: dict[str, float] = self._reward(response)
        self._update_state(response)
        
        outcome = EpisodeOutcome(episode_length=self.timestep + 1)
        if response.terminated:
            terminations = {a: True for a in self.agents}

        if self.timestep >= MAX_TIMESTEP:  # Max timesteps reached
            truncations = {a: True for a in self.agents}
        self.timestep += 1

        infos = {a: {} for a in self.agents}

        is_done = any(terminations.values()) or any(truncations.values())
        if is_done:
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
        n_drones = len(self.agents) + len(self.evaders)

        n_data_points = n_drones * 6  # x,y,z,xv,yv,zv for each drone

        n_agent_id = len(
            self.agents
        )  # one-hot agent ID, so drones can share a policy but still know who they are

        return Box(
            low=0.0, high=1.0, shape=(n_data_points + n_agent_id,), dtype=np.float32
        )

    def action_space(self, agent):
        low = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return Box(low=low, high=high, shape=(3,), dtype=np.float32)
