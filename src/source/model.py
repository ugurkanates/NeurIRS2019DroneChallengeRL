import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
@@ -11,17 +12,22 @@ def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor_hidden1 = nn.Linear(num_inputs, hidden_size)
        self.mu = nn.Linear(hidden_size, num_outputs)
        self.var = nn.Linear(hidden_size, num_outputs)

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        # self.actor = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, num_outputs),
        # )
        # self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
        x = F.relu(self.actor_hidden1(x))
        mean = torch.clamp(self.mu(x), -1.0, 1.0)
        var = F.softplus(self.var(x)) + 1e-5 # softplus clamps values between -1 and +1
        dist  = Normal(mean, var.sqrt())
        return dist, value 
