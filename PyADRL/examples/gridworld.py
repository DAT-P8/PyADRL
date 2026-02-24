import grpc
import random
import grid_world_pb2
from envs.gridworld_env import GridWorldEnvironment

# Connect to your gRPC server
channel = grpc.insecure_channel('localhost:50051')
env = GridWorldEnvironment(channel)

# Reset to initialize the environment
env.reset()

terminated = False
truncated = False

while not terminated and not truncated:
    # Build actions for each agent
    actions = []
    
    # Action for evader
    actions.append(grid_world_pb2.GWDroneAction(
        id=env.evader.id,
        action=random.randint(1, 5)
    ))
    
    # Actions for each pursuer
    for pursuer in env.pursuer:
        actions.append(grid_world_pb2.GWDroneAction(
            id=pursuer.id,
            action=random.randint(1, 5)
        ))
    
    pursuer_reward, evader_reward, terminated, truncated = env.step(actions)
    
    print(f"Step {env.timestep} | Pursuer reward: {pursuer_reward} | Evader reward: {evader_reward}")
    print(f"  Evader pos: ({env.evader.x}, {env.evader.y})")
    for p in env.pursuer:
        print(f"  Pursuer {p.id} pos: ({p.x}, {p.y})")