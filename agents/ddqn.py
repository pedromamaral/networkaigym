import gym
import torch
import numpy as np
from torch import nn
import random
import torch.nn.functional as F
import collections
from torch.optim.lr_scheduler import StepLR

"""
Implementation of Double DQN for gym environments with discrete action space.
source: https://github.com/fschur/DDQN-with-PyTorch-for-OpenAI-Gym/blob/master/DDQN_discrete.py
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
The Q-Network has as input a state s and outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
"""
class QNetwork(nn.Module):
    def __init__(self, action_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(845, 1500)
        self.fc_2 = nn.Linear(1500, 700)
        self.fc_3 = nn.Linear(700, 200)
        self.fc_4 = nn.Linear(200, action_dim)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = F.leaky_relu(self.fc_3(x1))
        x1 = self.fc_4(x1)

        return x1

"""
memory to save the state, action, reward sequence from the current episode. 
"""
class Memory:
    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.state = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.is_done)
        idx = random.sample(range(0, n-1), batch_size)

        return torch.Tensor(self.state)[idx].to(device), torch.LongTensor(self.action)[idx].to(device), \
               torch.Tensor(self.state)[1+np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done)[idx].to(device)

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()

def select_action_train(model, env, state, eps):
    _state = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, 845)
    state = _state.to(device)

    with torch.no_grad():
        values = model(state)

    # select a random action wih probability eps
    if random.random() <= eps:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = np.argmax(values.cpu().numpy())

    return action

def select_action_test(model, state):
     # _state = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, 845)
     with torch.no_grad():
         values = model(state)
     action = np.argmax(values.cpu().numpy())

     return action  

def train(batch_size, current, target, optim, memory, gamma):

    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    _states = states.reshape(256, 845)
    _next_states = next_states.reshape(256, 845)

    q_values = current(_states)

    next_q_values = current(_next_states)
    next_q_state_values = target(_next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

def evaluate(Qmodel, env, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()
    perform = 0
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            _state = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, 845).to(device)
            with torch.no_grad():
                values = Qmodel(_state)
            action = np.argmax(values.cpu().numpy())
            state, reward, done, _ = env.step(action)
            perform += reward
    Qmodel.train()
    return perform/repeats

def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def test_model(model, env):
    _state = env.reset()
    state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, 845)
    done = False
    rewards = []
    while not done:
        # DDQN
        action = select_action_test(model, state)
        
        _state, reward, done, _ = env.step(action)
        state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, 845)
        rewards.append(reward)
        print("Request:", len(rewards), "Path:", action, "Reward:", reward)
    print("Reward sum:", sum(rewards))


def ddqn_agent(gamma=0.99, lr=1e-3, min_episodes=20, eps=1, eps_decay=0.995, eps_min=0.01, update_step=10, batch_size=64, update_repeats=50,
         num_episodes=3000, seed=42, max_memory_size=50000, lr_gamma=0.9, lr_step=100, measure_step=100,
         measure_repeats=100, hidden_dim=64,  **env):
    """
    :param gamma: reward discount factor
    :param lr: learning rate for the Q-Network
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
    :param eps: probability to take a random action during training
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
    :param eps_min: minimal value of "eps"
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param update_repeats: see above
    :param num_episodes: the number of episodes played in total
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param lr_gamma: learning rate decay for the Q-Network
    :param lr_step: every "lr_step" episodes we decay the learning rate
    :param measure_step: every "measure_step" episode the performance is measured
    :param measure_repeats: the amount of episodes played in to asses performance
    :param hidden_dim: hidden dimensions for the Q_network
    :param env_name: name of the gym environment
    :return: 
    """
    total_rewards = []
    env = env
    torch.manual_seed(seed)
    env.seed(seed)

    Q_1 = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
                                        hidden_dim=hidden_dim).to(device)
    Q_2 = QNetwork(action_dim=env.action_space.n, state_dim=env.observation_space.shape[0],
                                        hidden_dim=hidden_dim).to(device)
        
    # transfer parameters from Q_1 to Q_2
    update_parameters(Q_1, Q_2)

    # we only train Q_1
    for param in Q_2.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    memory = Memory(max_memory_size)
    performance = []

    for episode in range(num_episodes):
        # display the performance
        if episode % measure_step == 0:
            performance.append([episode, evaluate(Q_1, env, measure_repeats)])
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            total_rewards.append(performance[-1][1])
            print("lr: ", scheduler.get_lr()[0])
            print("eps: ", eps)

        state = env.reset()
        memory.state.append(state)

        done = False
        i = 0
        while not done:
            i += 1
            action = select_action_train(Q_2, env, state, eps)
            state, reward, done, _ = env.step(action)

            #if i > horizon:
            #    done = True

            # render the environment if render == True
            #if render and episode % render_step == 0:
            #    env.render()

            # save state, action, reward sequence
            memory.update(state, action, reward, done)

        if episode >= min_episodes and episode % update_step == 0:
            for _ in range(update_repeats):
                train(batch_size, Q_1, Q_2, optimizer, memory, gamma)

            # transfer new parameter from Q_1 to Q_2
            update_parameters(Q_1, Q_2)

        # update learning rate and eps
        scheduler.step()
        eps = max(eps*eps_decay, eps_min)
    #Q_1, performance, total_rewards
    test_model(Q_1,env) 
    """
    sizes = [25, 50, 100, 200, 500]
    for size in sizes:
        avg = []
        for idx in range(0, len(total_rewards), size):
            avg += [sum(val for val in total_rewards[idx:idx + size]) / size]

        plt.figure(figsize=(10, 7))
        plt.plot(avg)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Return", fontsize=22)
        plt.savefig('ddqn_policy_set1_32r_4sync_{}.png'.format(size))

    """