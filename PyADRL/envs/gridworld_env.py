"""Gridworld MA area defense environment"""

from pettingzoo import ParallelEnv
import grpc
import grid_world_pb2, grid_world_pb2_grpc
from utils.gridworld_client import GridWorldClient
from gymnasium.spaces import Discrete, MultiDiscrete

class Drone:
    def __init__(self, id: int, x: int, y: int, is_evader: bool):
        self.id : int = id
        self.x = x
        self.y = y
        self.is_evader = is_evader

class GridWorldEnvironment(ParallelEnv):
    metadata = {
        "name": "gridworld_environment_v0",
    }

    def __init__(self, channel: grpc.Channel):
        self.id: int = None
        self.client = GridWorldClient(channel)
        self.target_x: int = 5
        self.target_y: int = 5
        self.pursuer: list[Drone] = []
        self.evader: Drone = None
        self.timestep: int = 0

    def reset(self, seed=None, options=None):
        state = None

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
                self.pursuer.append(Drone(drone_state.id, drone_state.x, drone_state.y, False))

    def step(self, actions : list[grid_world_pb2.GWDroneAction]):
        pursuer_reward = 0
        evader_reward = 0
        terminated = False
        truncated = False

        response = self.client.DoStep(grid_world_pb2.GWActionRequest(id=self.id, actions=actions))
        if response.WhichOneof("state_or_error")=="state":
            state = response.state
            for drone_state in state.drone_states:
                if drone_state.destroyed and drone_state.is_evader==False:
                    self.pursuer = [p for p in self.pursuer if p.id != drone_state.id]
                    pursuer_reward -= 10
                if drone_state.is_evader:
                    self.evader.x = drone_state.x
                    self.evader.y = drone_state.y
                else:
                    for pursuer in self.pursuer:
                        if pursuer.id == drone_state.id:
                            pursuer.x = drone_state.x
                            pursuer.y = drone_state.y
        else:
            raise ValueError("Error in step")
        

        if response.state.terminated:
            if self.evader.x == self.target_x and self.evader.y == self.target_y:
                evader_reward += 100
                terminated = True
                print("Evader reached target!")
            elif any(pursuer.x == self.evader.x and pursuer.y == self.evader.y for pursuer in self.pursuer):
                evader_reward -= 100
                pursuer_reward += 100
                terminated = True
                print("Evader caught by pursuer!")
            elif self.evader.x > 10 or self.evader.y > 10 or self.evader.x < 0 or self.evader.y < 0:
                evader_reward -= 10
                terminated = True
                print("Evader out of bounds!")

        if self.timestep >= 100:
            truncated = True
            print("Max timesteps reached!")

        pursuer_reward -= 1
        evader_reward -= 1

        self.timestep += 1

        return pursuer_reward, evader_reward, terminated, truncated

    def render(self):
        pass

    def observation_space(self, agent):
        return MultiDiscrete([11*11]*(2+len(self.pursuer)))

    def action_space(self, agent):
        return Discrete(5)  # 5 possible actions: up, down, left, right, stay
