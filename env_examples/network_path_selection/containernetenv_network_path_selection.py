import os
import gym
from gym import Env, spaces
import numpy as np
import copy
from time import sleep
import random
import envs.networkX_api_topo as netX

# example network path selection
from envs.containernet_api_topo import ContainernetAPI
from env_examples.network_path_selection.parameters import TOPOLOGY_FILE,DOCKER_VOLUME,NUMBER_HOSTS,NUMBER_PATHS,REWARD_SCALE

state_helper = {}
host_pairs = [('H4', 'H8'), ('H2', 'H11'), ('H2', 'H13'), ('H2', 'H9'), ('H4', 'H11'), ('H4', 'H9'), ('H2', 'H8'), ('H1', 'H11'),
             ('H1', 'H9'), ('H4', 'H13'), ('H4', 'H10'), ('H4', 'H7'), ('H3', 'H8'), ('H2', 'H10'), ('H2', 'H7'), ('H1', 'H8'), 
             ('H4', 'H12'), ('H3', 'H11'), ('H2', 'H12'), ('H1', 'H13'), ('H3', 'H9'), ('H1', 'H12'), ('H1', 'H7'), ('H4', 'H6'), 
             ('H3', 'H10'), ('H5', 'H6'), ('H3', 'H13'), ('H3', 'H7'), ('H7', 'H6'), ('H5', 'H11'), ('H5', 'H8'), ('H3', 'H12')]
busy_ports = [6631, 6633]

class ContainernetEnv(Env):
    def __init__(self):
        super(ContainernetEnv, self).__init__()

        self.containernet = ContainernetAPI(TOPOLOGY_FILE)

        self.graph = netX.build_graph_from_txt(self.containernet.bw_capacity)
        
        self.containernet.paths = netX.get_k_shortest_paths(self.graph, NUMBER_HOSTS, NUMBER_PATHS)

        self.upload_starting_rules()

        # define Observation Space

        self.observation_space = spaces.Box(
            low=np.zeros((NUMBER_HOSTS, NUMBER_HOSTS,
                         NUMBER_PATHS, 1), dtype=np.float32),
            high=np.full((NUMBER_HOSTS, NUMBER_HOSTS, NUMBER_PATHS, 1), 100, dtype=np.float32), dtype=np.float32)

        # define Action Space
        self.action_space = spaces.Discrete(NUMBER_PATHS)
        
        #define state
        self.state = build_state(self.containernet, NUMBER_HOSTS, NUMBER_PATHS)

        # define Env Elements/Variables
        self.number_of_requests = 0
        self.max_requests = 32
        

    ''' 
        get_state function

        translates the environment state into an observation
        The agent gets to a new state or observation state is
        the information of the environment that an agent is in
        and observation is an actual image that the agent sees.
    '''

    def get_state(self):
        return self.state

    ''' 
        get information function
        return auxiliary information that is returned by step and reset function
    '''

    def _get_info(self):
        print("_get_info function")

    ''' 
        reset function
    '''
    # reset function

    def reset(self):  
        self.done = False
        reset_measures(self.containernet)
        self.state = build_state(self.containernet, NUMBER_HOSTS, NUMBER_PATHS)
        self.number_of_requests = 0

        return self.state

    # render function
    def render(self, mode="human"):
        print("render function")

    # close function
    def close(self):
        print("close function")

    # step function
    def step(self, action):
        start_iperf_traffic(self.containernet,action)
        self.number_of_requests += 1
        
        sleep(5)
        
        reward = 0
        self.state = build_state(self.containernet, NUMBER_HOSTS, NUMBER_PATHS)
        
        for src in range(NUMBER_HOSTS):
            for dst in range(NUMBER_HOSTS):
                for path_number in range(NUMBER_PATHS):
                    bw = self.state[src, dst, path_number]
                    
                    link = get_state_helper().get(str(src + 1) + "_" + str(dst + 1) + "_" + str(path_number))
                    
                    if link:
                        ex_link = link.split("_")
                        bw_percentage = self.get_percentage(ex_link[0], ex_link[1], bw[0])
                        if bw_percentage is not None:
                            if bw_percentage > 75:
                                reward += 50
                            elif bw_percentage > 50:
                                reward += 30
                            elif bw_percentage > 25:
                                pass
                            elif bw_percentage > 0:
                                reward -= 10
                            else:
                                reward -= 100
                        
        if self.number_of_requests == self.max_requests:
            sleep(181)
            self.done = True
            
        return self.state, reward/REWARD_SCALE, self.done, {}

    def upload_starting_rules(self):
        ofp_match_params={}
        paths_hops={}
        paths={}
        for src_idx, src_host in enumerate(self.containernet.network.hosts):
            src_host=src_host.MAC()
            for dst_idx, dst_host in enumerate(self.containernet.network.hosts):
                dst_host=dst_host.MAC()
                if src_host != dst_host:

                    path = netX.dijkstra(self.graph, src_host, dst_host)
                    paths_hops[(src_host, dst_host)] = path
                    paths[(src_host, dst_host)] = netX.add_ports_to_path(path, self.containernet.adjacency, src_host, dst_host)
                    
                    if src_host not in ofp_match_params:
                        ofp_match_params[src_host]={}
                    ofp_match_params[src_host][dst_host]={}
                    ofp_match_params[src_host][dst_host]=self.containernet.define_ofp_match_params(src_host, dst_host)

        
        self.containernet.upload_data_in_json_file(paths_hops, "starting_rules/paths_hops.json", "w")
        self.containernet.upload_data_in_json_file(paths, "starting_rules/paths.json", "w")
        self.containernet.upload_data_in_json_file(ofp_match_params, "starting_rules/ofp_match_params.json", "w")
        
    def get_percentage(self, src, dst, bw):

        if self.containernet.bw_capacity.get((src, dst)):
            return (bw / self.containernet.bw_capacity.get((src, dst))) * 100
        else:
            return None
                    

#***************************************************************************************************************************************
#************************************************AUXILIAR FUNCTIONS*********************************************************************
#***************************************************************************************************************************************
# build the network state using the controller stats and paths dict
def build_state(containernet, n_hosts, n_paths):
    state = np.empty((n_hosts, n_hosts, n_paths, 1), dtype=object)

    for src in range(1, n_hosts+1):
        h_src=containernet.get_host_mac("H{}".format(src))
        for dst in range(1, n_hosts+1):
            h_dst=containernet.get_host_mac("H{}".format(dst))

            cnt = 0
            
            if len(containernet.paths[(h_src, h_dst)]) == 1:
                if not containernet.paths[(h_src, h_dst)]:
                    for idx in range(n_paths):
                        state[src-1, dst-1, idx] = -1
                else:
                    state[src-1, dst-1, 0] = 100
                    for idx in range(1, n_paths):
                        state[src-1, dst-1, idx] = -1
            else:
                for path in containernet.paths[(h_src, h_dst)]:
                    
                    min_value = float('Inf')
                    for s1, s2 in zip(path[:-1], path[1:]):
                        
                        stats = containernet.bw_available_cumulative[(str(s1), str(s2))]
                        if stats:
                            if float(stats) < float(min_value):
                                min_value = containernet.bw_available_cumulative[(str(s1), str(s2))]
                                state_helper[str(src) + "_" + str(dst) + "_" + str(cnt)] = str(s1) + "_" + str(s2)
                    
                    state[src-1, dst-1, cnt] = float(min_value)
                    cnt += 1
                    
                for idx in range(len(containernet.paths[(h_src, h_dst)]), n_paths):
                    state[src-1, dst-1, idx] = -1
    
    return state


def get_state_helper():
    return state_helper

# clear files and variables
def reset_measures(containernet):
    global host_pairs, busy_ports

    containernet.bw_available_cumulative = copy.deepcopy(containernet.bw_capacity)
    busy_ports = [6631, 6633]
    os.system("rm -f ./*.log")
    containernet.active_paths={}
    containernet.ofp_match_params={}
    
    host_pairs = [('H4', 'H8'), ('H2', 'H11'), ('H2', 'H13'), ('H2', 'H9'), ('H4', 'H11'), ('H4', 'H9'), ('H2', 'H8'), ('H1', 'H11'),
                  ('H1', 'H9'), ('H4', 'H13'), ('H4', 'H10'), ('H4','H7'), ('H3', 'H8'), ('H2', 'H10'), ('H2', 'H7'), ('H1', 'H8'),
                  ('H4', 'H12'), ('H3', 'H11'), ('H2', 'H12'), ('H1','H13'), ('H3', 'H9'), ('H1', 'H12'), ('H1', 'H7'), ('H4', 'H6'),
                  ('H3', 'H10'), ('H5', 'H6'), ('H3', 'H13'), ('H3', 'H7'), ('H7', 'H6'), ('H5', 'H11'), ('H5', 'H8'), ('H3', 'H12')]
    containernet.ofp_match_params={}
    
def start_iperf_traffic(containernet,action):
    hosts_pair = host_pairs.pop(0)
    
    client = containernet.get_host_mac(hosts_pair[0])
    server = containernet.get_host_mac(hosts_pair[1])
    
    path = containernet.paths[(client, server)][action]
    path_r = containernet.paths[(server, client)][action]
    
    containernet.send_path_to_controller(client,server ,path,path_r)
    containernet.get_bw_used_bw_available()
    
    while True:
        port = random.randint(1000, 4097)
        if port not in busy_ports: break
    duration=5
    bw=15
    traffic_type="tcp"
    containernet.generate_traffic_with_iperf(hosts_pair[0], hosts_pair[1],port, duration, bw,traffic_type)


