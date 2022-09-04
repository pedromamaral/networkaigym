import gym
from mininet.net import Containernet
from gym import Env, spaces


class ContainernetEnv(Env):
    def __init__(self):
        super(ContainernetEnv,self).__init__()

        #define Observation Space
        self.observation_shape
        self.observation_space
        #define Action Space
        self.action_space
        
        #define Env Elements/Variables
    
    ''' 
        get_state function
        translates the environment state into an observation
        The agent gets to a new state or observation state is
        the information of the environment that an agent is in
        and observation is an actual image that the agent sees.
    '''

    def get_state(self):

    ''' 
        get information function
        return auxiliary information that is returned by step and reset function
    '''
    def _get_info(self):
        print("_get_info function")

    ''' 
        reset function
    '''
    #reset function
    def reset(self):
        print("reset function")        

    #step function
    def step(self, action):
        print("step function")

