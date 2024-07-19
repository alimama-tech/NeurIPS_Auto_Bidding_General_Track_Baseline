import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np


class Actor(nn.Module):
    def __init__(self, dim_observation, hidden_size=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_observation, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs):
        return self.net(obs)


class BC(nn.Module):
    """
        Usage:
        bc = BC(dim_obs=16)
        bc.load_net(load_path="path_to_saved_model")
        actions = bc.take_actions(states)
    """

    def __init__(self, dim_obs, actor_lr=0.0001, network_random_seed=1, actor_train_iter=3):
        super().__init__()
        self.dim_obs = dim_obs
        self.actor_lr = actor_lr
        self.network_random_seed = network_random_seed
        torch.manual_seed(self.network_random_seed)
        self.actor = Actor(self.dim_obs)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.actor_train_iter = actor_train_iter
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.actor.to(self.device)
        self.train_episode = 0

    def step(self, states, actions):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        loss_list = []
        for _ in range(self.actor_train_iter):
            predicted_actions = self.actor(states)
            loss = nn.MSELoss()(predicted_actions, actions)
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            loss_list.append(loss.item())
        return np.array(loss_list)

    def take_actions(self, states):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions = self.actor(states)
        actions = actions.clamp(min=0).cpu().numpy()
        return actions

    def save_net_pkl(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "bc.pkl")
        torch.save(self.actor, file_path)

    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/bc_model.pth')

    def forward(self, states):

        with torch.no_grad():
            actions = self.actor(states)
        actions = torch.clamp(actions, min=0)
        return actions

    def save_net(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "bc.pt")
        torch.save(self.actor.state_dict(), file_path)

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cpu'):
        file_path = os.path.join(load_path, "bc.pt")
        self.actor.load_state_dict(torch.load(file_path, map_location=device))
        self.actor.to(self.device)
        print(f"Model loaded from {self.device}.")

    def load_net_pkl(self, load_path="saved_model/fixed_initial_budget"):
        file_path = os.path.join(load_path, "bc.pkl")
        self.actor = torch.load(file_path, map_location=self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.actor.to(self.device)
        print(f"Model loaded from {self.device}.")
