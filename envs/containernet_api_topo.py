from mininet.net import Containernet
from mininet.node import RemoteController, Host, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from os import system

import time
import random
import copy
import envs.networkX_api_topo as netX
import json
import subprocess as sp
import os

from envs.parameters import TOPOLOGY_FILE, DOCKER_VOLUME,LOG_TIMEOUT


"""
    get data and save it in a json file given a expecific name and open_model
"""


def upload_data_in_json_file(data, file_name, open_model):
   
    filename = DOCKER_VOLUME+file_name
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if open_model != "r" or open_model != "r+":
        with open(filename, open_model) as f:
            json.dump({str(k): v for k, v in data.items()}, f)
            f.close()


def json_from_log(client, server, port, node_type, traffic_type):
    data = {}
    start_time: time = time()
    current_time: time = time()
    while not data and current_time - start_time < LOG_TIMEOUT:
        try:
            if node_type == "server":
                with open(f"{DOCKER_VOLUME}logs/server_{server}_{client}_{port}.log", 'r') as f:
                    data = json.load(f)
            else:
                if traffic_type == "udp":
                    with open(f"{DOCKER_VOLUME}logs/client_{client}_{server}_{port}_udp.log", 'r') as f:
                        data = json.load(f)
                elif traffic_type == "tcp":
                    with open(f"{DOCKER_VOLUME}logs/client_{client}_{server}_{port}_tcp.log", 'r') as f:
                        data = json.load(f)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            time.sleep(0.2)
        current_time = time()
    return data


def get_data_from_json(file_name):
    data = {}
    filename=DOCKER_VOLUME+file_name
    while data=={}:
        try:
            with open(filename, 'r') as f:
        
                if os.stat(filename).st_size == 0:
                    return {}
                data = json.loads(f.read())
            f.close()
            return data
        except:
            pass


"""
    Get stats from json.log file
    example:data=json_from_log(***)
            param1="end"
            param2="sum"
            param3="bits_per_second"
            return data["end"]["sum"]["bits_per_second"]=1.42846e+07
"""


def get_traffic_stats(data, param1, param2, param3):
    if param1 and param2 and param3:
        return data[param1][param2][param3]
    elif param1 and param2:
        return data[param1][param2]
    elif param1:
        return data[param1]
    else:
        return None



class ContainernetAPI:
    def __init__(self, n_hosts, n_paths, init_costs=None):
        system('clear')
        system('sudo rm -rf '+DOCKER_VOLUME)
        system('sudo mn -c')
        self.clear_logs()
        self.network = Containernet(controller=RemoteController, switch=OVSSwitch, link=TCLink,
                                    autoSetMacs=True, ipBase='10.0.0.0/8')
        self.bw_capacity = {}
        self.host_ip_mac = {}

        self.n_hosts = n_hosts
        self.n_paths = n_paths

        self.bw_used = {}
        

        self.state_helper = {}
        self.controller_stats = {}
        self.ofp_match_params={}

        
        self.switches={}
        self.adjacency={}
        self.active_connections = {}
        self.device_intf = {}

        self.load_topology(TOPOLOGY_FILE)

        self.network.addController(
            'c0', controller=RemoteController, ip='127.0.0.1', port=6653)
        self.network.start()

        self.change_hosts_id_to_hosts_mac()    

        self.bw_available = copy.deepcopy(self.bw_capacity)

        upload_data_in_json_file(self.bw_capacity, "topology/bandwidth_links.json", "w")
        
        self.upload_hosts_ip_mac_switch_out_port()
        self.upload_sw_adjacency()
        self.add_arps()

        if init_costs:
            self.get_initial_costs(init_costs)
            self.graph = netX.build_graph_from_txt(self.bw_capacity,self.init_costs)
        else:
            self.graph = netX.build_graph_from_txt(self.bw_capacity)
        
        self.paths = netX.get_k_shortest_paths(self.graph, self.n_hosts, self.n_paths)
       

    def get_host_mac(self,host):
        return self.network.getNodeByName(host).MAC()

    def change_hosts_id_to_hosts_mac(self):

        for (device1, device2) in self.bw_capacity.copy():
            if "S" not in device1:
                node=self.network.getNodeByName(device1)
                self.bw_capacity[(node.MAC(), device2)]=self.bw_capacity.pop((device1,device2))
            
            if "S" not in device2:
                node=self.network.getNodeByName(device2)
                self.bw_capacity[(device1, node.MAC())]=self.bw_capacity.pop((device1,device2))

    def clear_logs(self):
        system(f'rm -f {DOCKER_VOLUME}*.log')

    def add_host(self, name, container_params):
        if name not in self.network.keys():
            system(f'sudo docker rm -f mn.{name}')
            
            self.network.addDocker(name=name, **container_params)

    def add_switch(self, name):
        if name not in self.network.keys():
            self.switches[name.replace("S", "")] = name
            self.network.addSwitch(name)

    def add_link(self, source, destination, link_options):
        if not self.network.linksBetween(self.network.get(source), self.network.get(destination)):
            self.network.addLink(self.network.get(
                source), self.network.get(destination), **link_options)

    def load_topology(self, file):
        
        with open(file, 'r') as topology:
            for line in topology.readlines():
                cols = line.split()
                for node in cols[:2]:
                    if node[0] == 'S':
                        self.add_switch(node)
                    else:
                        """
                            Need to define cpu Container settings too:
                            -> dimage = docker image
                            -> cpu_quota = Limit CPU CFS (Completely Fair Scheduler) quota
                            -> cpu_period = Limit CPU CFS (Completely Fair Scheduler) period
                            -> cpu_shares = CPU shares (relative weight)-to set the weight of the CPU used by the container
                            -> cpuset_cpus = CPUs in which to allow execution
                            -> volumes = shared volumes
                            ...(you can add more parameters - see Docker class on node.py)
                        """
                        if len(cols) < 6:
                            container_params = dict(dimage='iperf:latest', volumes=[
                                                    f'{DOCKER_VOLUME}logs:/home/volume'])
                        else:
                            container_params = dict(dimage='iperf:latest', cpu_quota=cols[5], cpu_period=cols[6],
                                                    cpu_shares=cols[7], cpuset_cpus=cols[8], volumes=[f'{DOCKER_VOLUME}logs:/home/volume'])
                       
                        self.add_host(node, container_params)

                link_bw = int(cols[2])
                if len(cols) < 4:
                    link_options = dict(bw=link_bw)
                else:
                    link_options = dict(
                        bw=link_bw, delay=f'{cols[3]}ms', loss=float(cols[4]))
                self.add_link(cols[0], cols[1], link_options)
                self.bw_capacity[(cols[0], cols[1])] = link_bw
                self.bw_capacity[(cols[1], cols[0])] = link_bw
        

    # define starting ARP rules
    def add_arps(self):
        
        for src_idx, src in enumerate(self.network.hosts):
            for dst_idx, dst in enumerate(self.network.hosts):
                if src_idx != dst_idx:
                    src.cmd(f'arp -s {dst.IP()} {dst.MAC()}')
                    

    # create traffic between src and dst
    def generate_traffic_with_iperf(self, source, destination, port, duration, bw, traffic_type):
        src = self.network.get(source)
        dst = self.network.get(destination)
        
        print("iperf",src,dst,bw)

        if src and dst:
            dst.cmd(
                f'iperf3 -s -p {port} -i 1 -J >& /home/volume/server_{destination}_{source}_{port}.log &')
            if traffic_type == 'udp':
                src.cmd(
                    f'iperf3 -c {dst.IP()} -p {port} -t {duration} -b {bw}M -J -u >& /home/volume/client_{source}_{destination}_{port}_udp.log &')
            elif traffic_type == "tcp":
                src.cmd(
                    f'iperf3 -c {dst.IP()} -p {port} -t {duration} -b {bw}M -J >& /home/volume/client_{source}_{destination}_{port}_tcp.log &')
            time.sleep(1)

    def get_bw_used_bw_available(self):

        for (device1, device2) in self.bw_capacity:
            if "S" in device1:
                s1_dpid = device1.replace("S", "")

                bw_json = get_data_from_json(
                    "switches/sw"+str(s1_dpid)+"/bandwidth_sw"+str(s1_dpid)+".json")
               
                if bw_json !={}:
                    for s1_id,value in bw_json.items():
                        for s2_id in value:
                            if bw_json[s1_id][s2_id].get("bandwidth_used") != {}:
                                self.bw_used[("S"+str(s1_id), "S"+str(s2_id))] = bw_json[s1_id][s2_id].get("bandwidth_used")

                            if bw_json[s1_id][s2_id].get("bandwidth_available") != {}:
                                #self.bw_available[("S"+str(s1_id), "S"+str(s2_id))] = bw_json[s1_id][s2_id].get("bandwidth_available")
                                self.bw_available[("S"+str(s1_id), "S"+str(s2_id))] =float(self.bw_available[("S"+str(s1_id), "S"+str(s2_id))]) \
                                    -float(bw_json[s1_id][s2_id].get("bandwidth_used"))
                                
                            

        
                
    """
        all the parameters that can be used in ofp_match are in
        in the link: https://ryu.readthedocs.io/en/latest/ofproto_v1_3_ref.html#flow-match-structure
    """

    def define_ofp_match_params(self, client, server):
        ofp_match_params = {}
        
        
        ofp_match_params["eth_src"] = client
        ofp_match_params["eth_dst"] = server
        return ofp_match_params

    # send paths to the controller for rule installation
    def send_path_to_controller(self, action, client, server):

        client = self.get_host_mac(client)
        server = self.get_host_mac(server)
        
        if client not in self.ofp_match_params:
            self.ofp_match_params[client]={}
        self.ofp_match_params[client][server]={}
        self.ofp_match_params[client][server]=self.define_ofp_match_params(client, server)

        if server not in self.ofp_match_params:
            self.ofp_match_params[server]={}
        self.ofp_match_params[server][client]={}
        self.ofp_match_params[server][client]=self.define_ofp_match_params(server, client)

        upload_data_in_json_file(
            self.ofp_match_params, "OFPMatch/OFPMatch_params.json", "w")


        path = self.paths[(client, server)][action]
        path_r = self.paths[(server, client)][action]
        

        try:
            f = open(DOCKER_VOLUME+"active_paths.txt", "a")
            f.write('{}_{}_{} \n'.format(server, client, path_r))
            f.write('{}_{}_{} \n'.format(client, server, path))
            f.close()
        except IOError:
            print("file not ready")

        time.sleep(1)

        self.get_bw_used_bw_available()

    def get_percentage(self, src, dst, bw):
        
        if "H" not in str(src) and "S" not in str(src):
            src_str = "S" + str(src)
        else:
            src_str = src

        if "H" not in str(dst) and "S" not in str(dst):
            dst_str = "S" + str(dst)
        else:
            dst_str = dst

        if self.bw_capacity.get((src_str, dst_str)):
            return (bw / self.bw_capacity.get((src_str, dst_str))) * 100
        else:
            return None

    """
        Resource limitations based on CFS scheduler:
        * cpu_quota: the total available run-time within a period (in microseconds)
        * cpu_period: the length of a period (in microseconds)
        * cpu_shares: Relative amount of max. avail CPU for container
        (not a hard limit, e.g. if only one container is busy and the rest idle)
        e.g. usage: d1=4 d2=6 <=> 40% 60% CPU

        -> As the default value of cpu-period is 100000, setting the value of 
        cpu-quota to 25000 limits a container to 25% of the CPU resources. 
            By default, a container can use all available CPU resources, which corresponds
        to a --cpu-quota value of -1
        
    """

    def update_container_cpu_limits(self, hostname, cpu_quota, cpu_period, cpu_shares, cores):
        host = self.network.getNodeByName(hostname)

        host.updateCpuLimit(cpu_quota, cpu_period, cpu_shares, cores)

    def add_container_to_topo(self, name, sw, ip, link_bw, delay, loss, **container_params):
        info('*** Dynamically add a container at runtime\n')
        container = self.add_host(name, container_params)
        # we have to specify a manual ip when we add a link at runtime
        ip_int = {"ip": ip}
        link_options = dict(bw=link_bw, delay=delay, loss=loss, ipint=ip_int)
        self.add_link(container, sw, link_options=link_options)

    """
    {container:mn.d1, memory:{raw:3.844MiB / 7.741GiB, percent:0.05%},cpu:0.00%}
    """

    def get_container_stats(self, name):
        container_name = 'mn.'+name
        output = sp.getoutput('docker stats '+container_name +
                              ' --no-stream --format "{container:{{.Container }}, memory:{raw:{{.MemUsage}}, percent:{{ .MemPerc }}},cpu:{{.CPUPerc}}}"')
        return output

    """
        save host ip mac and port out to sw in json file named hosts.json
    """

    def upload_hosts_ip_mac_switch_out_port(self):
        for i, node in enumerate(self.network.hosts):
            
            hostname = node.name
            self.host_ip_mac[node.MAC()]={}
            self.host_ip_mac[node.MAC()]['ip'] = node.IP()
            
            device_intf=self.get_device_intf_and_link(node)
            
            for sw in device_intf[str(node)]:
                port_out=int(device_intf[hostname][sw]['port_out'].replace("eth", ""))
                port_out=str(port_out+1)
                self.host_ip_mac[node.MAC()][sw.replace("S", "")] = str("eth"+port_out)
        upload_data_in_json_file(self.host_ip_mac, "topology/hosts.json", "w")
    
    """
        save adjacency - dict[(switch1,switch2),switch1_port_out]

    """
    def upload_sw_adjacency(self):
        for i, switch in self.switches.items():
            node = self.network.get(switch)
            device_intf=self.get_device_intf_and_link(node)
            
            for sw2 in device_intf[str(node)]:
                if "S" in sw2:
                    self.adjacency[(str(switch.replace("S","")),str(sw2.replace("S","")))]= int(device_intf[switch][sw2]['port_out'].replace("eth", ""))
        upload_data_in_json_file(self.adjacency, "topology/switches_adjacency.json", "w")
    
    """
        return devices and interface links of one specific device
        d1:{'s1': ('eth0', 'eth1'), 's2': ('eth1', 'eth1'), 's3': ('eth2', 'eth1')}
    """
    def get_device_intf_and_link(self,node):
        device_intf={}
        for intf in node.intfList():
            if intf.link:
                intfs = [intf.link.intf1, intf.link.intf2]
                intfs.remove(intf)
                src = str(intf).split("-")[0]
                port_out = str(intf).split("-")[1]
                dst = str(intfs[0]).split("-")[0]
                port_in = str(intfs[0]).split("-")[1]
                if not device_intf.get(src):
                    device_intf[src] = {}
                device_intf[src][dst] = {'port_out': port_out, 'port_in': port_in}
        
        return device_intf
    