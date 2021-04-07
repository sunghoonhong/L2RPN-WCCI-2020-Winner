import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from .sublayer import MultiHeadAttention, PositionwiseFeedForward


class GATLayer(nn.Module):
    def __init__(self, output_dim, nheads, node, action_dim, dropout=0):
        super(GATLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(nheads, output_dim, int((output_dim/4)), dropout=dropout) 
        self.pos_ffn = PositionwiseFeedForward(output_dim, output_dim, node, dropout=dropout)
    
    def forward(self, x, adj):
        x = self.slf_attn(x, adj)
        x = self.pos_ffn(x)
        return x
        
        
class Actor(nn.Module):
    def __init__(self, output_dim, nheads, node, action_dim, use_order, n_sub, dropout=0, init_w=3e-3, log_std_min=-10, log_std_max=1):
        super(Actor, self).__init__()
        self.gat1 = GATLayer(output_dim, nheads, node, action_dim, dropout)
        self.gat2 = GATLayer(output_dim, nheads, node, action_dim, dropout)
        self.gat3 = GATLayer(output_dim, nheads, node, action_dim, dropout)
        self.n_sub = n_sub
        self.use_order = use_order
        self.down = nn.Linear(output_dim, 1)
        self.mu = nn.Linear(int(node), int(action_dim))
        self.log_std = nn.Linear(int(node), int(action_dim))
        if use_order:
            self.order_mu = nn.Linear(node + action_dim, n_sub)
            self.order_log_std = nn.Linear(node + action_dim, n_sub)
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    def forward(self, x, adj):
        x = self.gat1(x, adj)
        x = self.gat2(x, adj) 
        x = self.gat3(x, adj)
        x = self.down(x).squeeze(-1)
        state = x
        x = F.leaky_relu(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mu, log_std, state
    
    def mean(self, x, adj):
        mu, _, state = self.forward(x, adj)
        action = torch.tanh(mu)
        if self.use_order:
            state = torch.tanh(state)
            s_a = torch.cat([state, action], dim=1)
            order = torch.tanh(self.order_mu(s_a))
            return action, order
        return action

    def sample(self, x, adj):
        mu, log_std, state = self.forward(x, adj)
        std = log_std.exp()
        normal = Normal(mu, std)
        z = normal.sample()
        action = torch.tanh(z)
        if self.use_order:
            state = torch.tanh(state)
            s_a = torch.cat([state, action], dim=1)
            order_mu = self.order_mu(s_a)
            order_log_std = self.order_log_std(s_a)
            order_log_std = torch.clamp(order_log_std, self.log_std_min, self.log_std_max)
            order_std = order_log_std.exp()
            order_normal = Normal(order_mu, order_std)
            order_z = order_normal.sample()
            order = torch.tanh(order_z)
            return (action, order), (std, order_std)
        return action, std
    
    
class EncoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim, nheads, node, action_dim, dropout=0):
        super(EncoderLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gat1 = GATLayer(output_dim, nheads, node, action_dim, dropout)
        self.gat2 = GATLayer(output_dim, nheads, node, action_dim, dropout)
        self.gat3 = GATLayer(output_dim, nheads, node, action_dim, dropout)
        self.gat4 = GATLayer(output_dim, nheads, node, action_dim, dropout)
        self.gat5 = GATLayer(output_dim, nheads, node, action_dim, dropout)
        self.gat6 = GATLayer(output_dim, nheads, node, action_dim, dropout)

    def forward(self, x, adj):
        x = self.linear(x)
        x = self.gat1(x, adj)
        x = self.gat2(x, adj)
        x = self.gat3(x, adj)
        x = self.gat4(x, adj)
        x = self.gat5(x, adj)
        x = self.gat6(x, adj)
        return x
