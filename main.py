from agents.ddqn import ddqn_agent
from agents.dqn import dqn_agent
from agents.duelingdqn import dueling_dqn_agent
from envs.containernetenvdiogo import ContainernetEnv
if __name__ == "__main__":
    #env= gym.make("containernet_gym/ContainernetEnv-v0")
    env=ContainernetEnv()
    #ddqn_agent(env=env)
    dqn_agent(env=env)
    #dueling_dqn_agent(env=env)