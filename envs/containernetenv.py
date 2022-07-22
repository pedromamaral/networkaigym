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
        get observation function
        translates the environment state into an observation
    '''
    def _get_obs(self):
        print("_get_obs function")

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

    #render function
    def render(self, mode = "human"):
        print("render function") 

    #close function
    def close(self):
        print("close function") 

    #step function
    def step(self, action):
        print("step function")

