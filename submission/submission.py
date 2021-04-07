import os
import numpy as np
import torch
import json
from .agent import Agent

def make_agent(env, submission_dir):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your sudmission directory and return a valid agent.
    """
    data_dir = os.path.join(submission_dir, 'model')
    with open(os.path.join(data_dir, 'param.json'), 'r', encoding='utf-8') as f:
        param = json.load(f)
    state_mean = torch.load(os.path.join(data_dir, 'mean.pt'), map_location=param['device']).cpu()
    state_std = torch.load(os.path.join(data_dir, 'std.pt'), map_location=param['device']).cpu()
    res = Agent(env, state_mean, state_std, **param)
    res.load_model(data_dir)
    return res
