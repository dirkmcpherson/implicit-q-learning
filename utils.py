import time
import numpy as np
import torch
from IPython import embed

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.size

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_dataset_from_end(self, num_samples):
        ind = np.arange(self.size - num_samples, self.size)

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )

    def add_dataset(self, dataset, num_samples):
        self.state[0:num_samples] = dataset[0]
        self.action[0:num_samples] = dataset[1]
        self.next_state[0:num_samples] = dataset[2]
        self.reward[0:num_samples] = dataset[3]
        self.not_done[0:num_samples] = dataset[4]
        

    # def remove(self, n):
    #     '''
    #     remove the first n entries from the buffer
    #     '''
    #     state = self.state 
    #     action = self.action 
    #     next_state = self.next_state 
    #     reward = self.reward 
    #     not_done = self.not_done 
    #     tmp_ptr = self.ptr

    #     self.ptr = 0
    #     self.size = 0
    #     self.state = np.zeros((self.max_size, self.state_dim))
    #     self.action = np.zeros((self.max_size, self.action_dim))
    #     self.next_state = np.zeros((self.max_size, self.state_dim))
    #     self.reward = np.zeros((self.max_size, 1))
    #     self.not_done = np.zeros((self.max_size, 1))
    #     # very slow
    #     for i in range(n):
    #         self.add(state[i+1], action[i+1], next_state[i+1], reward[i+1], not_done[i+1])

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def sample_normalized(self, batch_size, mean=0, std=1):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor((self.state[ind] - mean) / std).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor((self.next_state[ind] - mean) / std).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def convert_D4RL_pybullet(self, dataset):
        self.state = dataset['observations'][0:-1]
        self.action = dataset['actions'][0:-1]
        self.next_state = dataset['observations'][1:]
        self.reward = dataset['rewards'][0:-1].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'][0:-1].reshape(-1, 1)
        self.size = self.state.shape[0]

    def convert_npz(self, dataset):
        # for key in dataset.keys(): print(key)
        states = np.concatenate((dataset['states'], dataset["desired_goals"]), axis=1)
        self.state = states[0:-1]
        self.next_state = states[1:]
        self.action = dataset['actions'][0:-1]
        self.reward = dataset["rewards"][0:-1].reshape(-1, 1)
        self.not_done = 1. - dataset["dones"][0:-1].reshape(-1, 1)
        self.size = self.state.shape[0]
        self.max_size = self.size # just set the max size to be the size of the giant saved dataset #HACK
        self.ptr = 0

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std

    def calculate_meanstd(self,  eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        return mean, std