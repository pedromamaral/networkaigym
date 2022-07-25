from agents.ddqn import ddqn_agent
from agents.dqn import dqn_agent
from agents.duelingdqn import dueling_dqn_agent
from env_examples.dynamic_network_slicing.containernetenv_dynamic_network_slicing import ContainernetEnv
from env_examples.dynamic_network_slicing.parameters import INPUT_DIM,HL1, HL2, HL3, OUTPUT_DIM, GAMMA, EPSILON, LEARNING_RATE,EPOCHS, MEM_SIZE, BATCH_SIZE, SYNC_FREQ
#from env_examples.network_path_selection.containernetenv_network_path_selection import ContainernetEnv
if __name__ == "__main__":
    #env= gym.make("containernet_gym/ContainernetEnv-v0")
    env=ContainernetEnv()
    
    dqn_agent(gamma = GAMMA, epsilon = EPSILON,env=env,state_flattened_size=INPUT_DIM,l1 = INPUT_DIM, l2 = HL1, l3 = HL2,l4 = HL3, l5 = OUTPUT_DIM)
    #dqn_agent(env=env)

    #dueling_dqn_agent(env=env)
    #ddqn_agent(env=env)