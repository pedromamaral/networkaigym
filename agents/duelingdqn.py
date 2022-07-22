import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from itertools import count
import torch.nn.functional as F


"""
    source:https://github.com/gouxiangchen/dueling-DQN-pytorch/blob/master/dueling_dqn.py
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(845, 1500)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1500, 700)
        self.fc_value = nn.Linear(700, 200)
        self.fc_adv = nn.Linear(700, 200)

        self.value = nn.Linear(200, 1)
        self.adv = nn.Linear(200, 5)

    def forward(self, state):
        y = self.relu(self.fc1(state))
        y = self.relu(self.fc2(y))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        return Q

    def select_action(self, state):
        with torch.no_grad():
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()

class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

def test_model(model, env):
    _state = env.reset()
    state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, 845)
    done = False
    rewards = []
    while not done:
       
        # Dueling DQN
        action = model.select_action(state)
        _state, reward, done, _ = env.step(action)
        state = torch.flatten(torch.from_numpy(_state.astype(np.float32))).reshape(1, 845)
        rewards.append(reward)
        print("Request:", len(rewards), "Path:", action, "Reward:", reward)
    print("Reward sum:", sum(rewards))

def dueling_dqn_agent(gamma=0.99, final_eps=0.0001, eps=0.5, explore=20000, update_step=16, batch=256,
         seed=42, max_memory_size=50000, **env):
    """
    :param gamma: reward discount factor

    :param eps: probability to take a random action during training
    
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param env_name: name of the gym environment
    :return: 
    """
    total_rewards = []
    env = env
    
    torch.manual_seed(seed)
    env.seed(seed)
    
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    
    onlineQNetwork = QNetwork().to(device)
    targetQNetwork = QNetwork().to(device)
    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
    
    optimizer = torch.optim.Adam(onlineQNetwork.parameters(), lr=1e-4)

    memory_replay = Memory(max_memory_size)

    epsilon = eps
    learn_steps = 0
    begin_learn = False

    episode_reward = 0
    total_rewards = []

    for epoch in range(4000):
        state = env.reset()
        _state = torch.flatten(torch.from_numpy(state.astype(np.float32))).reshape(1, n_state)
        episode_reward = 0
        for time_steps in range(200):
            p = random.random()
            if p < epsilon:
                action = random.randint(0, n_action - 1)
            else:
                tensor_state = _state.to(device)
                action = onlineQNetwork.select_action(tensor_state)
            next_state, reward, done, _ = env.step(action)
            _next_state = torch.flatten(torch.from_numpy(next_state.astype(np.float32))).reshape(1, n_state)
            episode_reward += reward
            memory_replay.add((_state, _next_state, action, reward, done))
            if memory_replay.size() > batch:
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True
                learn_steps += 1
                if learn_steps % update_step == 0:
                    targetQNetwork.load_state_dict(onlineQNetwork.state_dict())
                batch = memory_replay.sample(batch, False)
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

                batch_state = torch.cat([item for item in batch_state]).to(device)
                batch_next_state = torch.cat([item for item in batch_next_state]).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

                with torch.no_grad():
                    onlineQ_next = onlineQNetwork(batch_next_state)
                    targetQ_next = targetQNetwork(batch_next_state)
                    online_max_action = torch.argmax(onlineQ_next, dim=1, keepdim=True)
                    y = batch_reward + (1 - batch_done) * gamma * targetQ_next.gather(1, online_max_action.long())

                loss = F.mse_loss(onlineQNetwork(batch_state).gather(1, batch_action.long()), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epsilon > final_eps:
                    epsilon -= (eps - final_eps) / explore

            if done:
                total_rewards.append(episode_reward)
                break
            _state = _next_state

        if epoch % 10 == 0:
            #torch.save(onlineQNetwork.state_dict(), 'dueling-dqn_policy_set1_32r_16sync.pt')
            print('Ep {}\tMoving average score: {:.2f}\t'.format(epoch, episode_reward))
    
    test_model(onlineQNetwork,env)

    """
    sizes = [100, 200, 500, 1000]
    for size in sizes:
        avg = []
        for idx in range(0, len(total_rewards), size):
            avg += [sum(val for val in total_rewards[idx:idx + size]) / size]

        plt.figure(figsize=(10, 7))
        plt.plot(avg)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Return", fontsize=22)
        plt.savefig('dueling-dqn_policy_set1_32r_16sync_{}.png'.format(size))
    """

