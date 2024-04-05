import torch
import torch.nn as nn
import random
import numpy as np
import gym
import math
from collections import namedtuple
from torchvision import transforms as T
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# Agent Training
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class DQN(nn.Module):
    def __init__(self, input_channels, output_dim) -> None:
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU()
        self.linears = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
        )
        self.Advantages = nn.Linear(512, output_dim)
        self.Value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.linears(x)
        value = self.Value(x)
        advantages = self.Advantages(x)
        return value, advantages

class Agent:
    def __init__(self) -> None:
        self.input_channels = 4
        self.n_actions = 12
        self.gamma = .9
        self.num_stack = 4
        self.device = torch.device("cpu")
        self.length_of_side_of_square = 84
        self.stack_frames = None

        self.learning_nn = DQN(input_channels=self.input_channels, output_dim=self.n_actions).to(self.device)
        self.target_nn = DQN(input_channels=self.input_channels, output_dim=self.n_actions).to(self.device)

        state_dict = torch.load('110062371_hw2_data.py')
        self.learning_nn.load_state_dict(state_dict=state_dict)
        self.target_nn.load_state_dict(state_dict=state_dict)


    def max_act(self, state:np.ndarray):

        if random.random() <= .01:
            action = random.randint(0, self.n_actions - 1)
        else:
            state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            _, action_vals = self.learning_nn(state)
            action = torch.argmax(action_vals, axis=1).item()

        return action
    
    def proportional_act(self, state:np.ndarray):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        _, action_vals = self.learning_nn(state)
        action_vals = torch.exp(action_vals)
        action_vals = action_vals.squeeze()
        sum_action_vals = torch.sum(action_vals).item()

        random_val = random.random() * sum_action_vals
        for idx, action_val in enumerate(action_vals):
            if random_val <= action_val.item():
                return idx
            random_val -= action_val.item()

        assert 1==0, 'can not be here {} {}'.format(random_val, action_val)
    
    def fill_stack_frames(self, state:np.ndarray):
        self.stack_frames = np.zeros((self.num_stack, state.shape[0], state.shape[1]))

        for i in range(self.num_stack):
            self.stack_frames[i] = state.copy()

    def add_frame(self, state:np.ndarray):
        for i in range(self.num_stack - 1):
            self.stack_frames[i] = self.stack_frames[i + 1]
        self.stack_frames[self.num_stack - 1] = state.copy()

    def outer_frame_to_inner_frame(self, state:np.ndarray):
        state = np.transpose(state, (2, 0, 1))
        state = torch.tensor(state.copy(), dtype=torch.float)
        transformation = T.Compose([
            T.Grayscale(), T.Resize((self.length_of_side_of_square, self.length_of_side_of_square), antialias=True), T.Normalize(0, 255)
        ])

        state = transformation(state).squeeze(0).__array__()

        return state

    def act(self, observation:np.ndarray):
        state = self.outer_frame_to_inner_frame(state=observation)

        if self.stack_frames is None:
            self.fill_stack_frames(state=state)
        else:
            self.add_frame(state=state)

        return self.max_act(state=self.stack_frames)


# frame_to_skip = 4
# length_of_side_of_square = 84
# num_stack = 4
# batch_size = 256
# dir_path = ''
# dir_agent_name = '110062371_hw2_data.py'

# device = torch.device("cpu")

# env = gym_super_mario_bros.make('SuperMarioBros-v0')
# env = JoypadSpace(env, COMPLEX_MOVEMENT)


# state_dict = torch.load(dir_path + dir_agent_name)
# agent = agent = Agent(input_channels=num_stack, length_of_side_of_square=length_of_side_of_square, n_actions=env.action_space.n, dir_name=dir_path + dir_agent_name, gamma=.9, num_stack=num_stack, device=device, state_dict=state_dict)


# done = True
# cu_reward = 0.
# for step in range(50):
#     if done:
#         state = env.reset()
    

#     count = 0
#     while True:
#         if count == 0:
#             action = agent.act(state=state)
#         state, reward, done, info = env.step(action)
#         env.render()
#         cu_reward += reward

#         count = (count + 1) % frame_to_skip

#         if done:
#             break
    
# print(cu_reward/50.)
