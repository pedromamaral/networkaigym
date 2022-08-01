
TOPOLOGY_FILE = '/home/pcapelo/Desktop/ai_gym_for_networks/env_examples/network_path_selection/topology_arpanet.txt'
DOCKER_VOLUME = '/home/pcapelo/Desktop/ai_gym_for_networks/volume/'
NUMBER_SWITCHES = 20
NUMBER_HOSTS = 13
NUMBER_PATHS = 5
REWARD_SCALE = NUMBER_HOSTS*NUMBER_HOSTS*NUMBER_PATHS
LOG_TIMEOUT = 90

INPUT_DIM = NUMBER_HOSTS * NUMBER_HOSTS * NUMBER_PATHS 
HL1: int = 1500
HL2: int = 700
HL3: int = 200
OUTPUT_DIM: int = NUMBER_PATHS

GAMMA: float = 0.99
EPSILON: float = 0.5
LEARNING_RATE: float = 1e-3

EPOCHS: int = 4000
MEM_SIZE: int = 50000
BATCH_SIZE: int = 256
SYNC_FREQ: int = 16

  