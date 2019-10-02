import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor_hidden1 = nn.Linear(num_inputs, hidden_size)
        self.mu = nn.Linear(hidden_size, num_outputs)
        self.var = nn.Linear(hidden_size, num_outputs)
        
        # self.actor = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, num_outputs),
        # )
        # self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
    def forward(self, x):
        value = self.critic(x)
        x = F.relu(self.act_hidden1(x))
        mean = torch.clamp(self.mu(x), -1.0, 1.0)
        var = F.softplus(self.std(x)) + 1e-5 # softplus clamps values between -1 and +1
        dist  = Normal(mean, var.sqrt())
        return dist, value
