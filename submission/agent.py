import os
import time
import random
import numpy as np
import torch
from .models import EncoderLayer, Actor
from .converter import graphGoalConverter
from grid2op.Agent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, env, state_mean, state_std, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        state_std = state_std.masked_fill(state_std < 1e-5, 1.)
        state_mean[0, sum(self.obs_space.shape[:20]):] = 0
        state_std[0, sum(self.obs_space.shape[:20]):] = 1
        self.state_mean = state_mean
        self.state_std = state_std
        super(Agent, self).__init__(env.action_space)
        
        mask = kwargs.get('mask')
        mask_hi = kwargs.get('mask_hi')
        self.danger = kwargs.get('danger')
        self.bus_thres = kwargs.get('threshold')
        self.max_low_len = kwargs.get('max_low_len')
        
        self.converter = graphGoalConverter(env, mask, mask_hi, self.danger, self.device)
        self.thermal_limit = env._thermal_limit_a
        self.convert_obs = self.converter.convert_obs
        self.action_dim = self.converter.n
        self.order_dim = len(self.converter.masked_sorted_sub)
        self.node_num = env.dim_topo

        self.nheads = kwargs.get('head_number')
        self.use_order = kwargs.get('use_order')
        self.dropout = kwargs.get('dropout')
        self.actor_lr = kwargs.get('actor_lr')
        self.critic_lr = kwargs.get('critic_lr')
        self.embed_lr = kwargs.get('embed_lr')
        self.alpha_lr = kwargs.get('alpha_lr')
        self.state_dim = kwargs.get('state_dim')
        self.n_history = kwargs.get('n_history')
        self.sim_trial = kwargs.get('sim_trial')

        self.input_dim = self.converter.n_feature * self.n_history
        print('O:', self.input_dim, 'S:', self.state_dim, 'A:', self.action_dim)
        
        self.emb = EncoderLayer(self.input_dim, self.state_dim, self.nheads,
                            self.node_num, self.action_dim, self.dropout).to(self.device)
        self.actor = Actor(self.state_dim, self.nheads, self.node_num, self.action_dim,
                            self.use_order, self.order_dim, self.dropout).to(self.device)
        self.emb.eval()
        self.actor.eval()
        
    def is_safe(self, obs):
        for ratio, limit in zip(obs.rho, self.thermal_limit):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= self.danger-0.05) or ratio >= self.danger:
                return False
        return True

    def state_normalize(self, s):
        s = (s - self.state_mean) / self.state_std
        return s

    def reset(self, obs):
        self.goal = None
        self.low_len = -1
        self.adj = None
        self.stacked_obs = []
        self.low_actions = []

    def stack_obs(self, obs):
        obs_vect = torch.FloatTensor(obs.to_vect()).unsqueeze(0)
        obs_vect = self.convert_obs(self.state_normalize(obs_vect))
        if len(self.stacked_obs) == 0:
            for _ in range(self.n_history):                
                self.stacked_obs.append(obs_vect)
        else:
            self.stacked_obs.pop(0)
            self.stacked_obs.append(obs_vect)
        self.adj = (torch.FloatTensor(obs.connectivity_matrix()) + torch.eye(int(obs.dim_topo))).to(self.device)

    def reconnect_line(self, obs):
        dislines = np.where(obs.line_status == False)[0]
        for i in dislines:
            if obs.time_next_maintenance[i] != 0 and i in self.converter.lonely_lines:
                sub_or = self.action_space.line_or_to_subid[i]
                sub_ex = self.action_space.line_ex_to_subid[i]
                if obs.time_before_cooldown_sub[sub_or] == 0:
                    return self.action_space({'set_bus': {'lines_or_id': [(i, 1)]}})
                if obs.time_before_cooldown_sub[sub_ex] == 0:
                    return self.action_space({'set_bus': {'lines_ex_id': [(i, 1)]}})
                if obs.time_before_cooldown_line[i] == 0:
                    status = self.action_space.get_change_line_status_vect()
                    status[i] = True
                    return self.action_space({'change_line_status': status})
        return None

    def get_current_state(self):
        return torch.cat(self.stacked_obs, dim=-1)

    def act(self, obs, reward, done):
        sample = False  
        self.stack_obs(obs)
        is_safe = self.is_safe(obs)
        if False in obs.line_status:
            act = self.reconnect_line(obs)
            if act is not None:
                return act
        if self.goal is None or (not is_safe and self.low_len == -1):
            _, goal, bus_goal, low_actions, order = self.generate_goal(sample, obs)
            if len(low_actions) == 0:
                if self.goal is None:
                    self.update_goal(goal, bus_goal, low_actions, order)
                return self.action_space()
            self.update_goal(goal, bus_goal, low_actions, order)
        act = self.rechoose_low_action(obs)
        return act
    
    def rechoose_low_action(self, obs):
        act = self.pick_low_action(obs)
        if self.sim_trial == 0:
            return act
        else:
            obs_s = obs.simulate(act)[0]
            # if current action yield blackout, try to generate new goal
            if type(obs_s) == type(None):
                success = False
                stacked_state = self.get_current_state().to(self.device)
                adj = self.adj.unsqueeze(0)
                _, temp_goal, _, temp_order = self.make_candidate_goal(stacked_state, adj, False, obs)
                candidate_goals = [temp_goal]
                candidate_orders = [temp_order]
                dn_tried = False
                for goal, order in zip(candidate_goals, candidate_orders):
                    low_actions = self.converter.plan_act(goal, obs, order)
                    low_actions = self.optimize_low_actions(obs, low_actions)
                    new_act = self.action_space() if len(low_actions) == 0 \
                            else self.converter.convert_act(*low_actions[0][:2])
                    if new_act == self.action_space():
                        if dn_tried:
                            # 'do nothing' has been already tried and failed, not worthy to try again.
                            continue
                        else:
                            dn_tried = True
                    obs_s = obs.simulate(new_act)[0]
                    # Got available goal!
                    if type(obs_s) != type(None):
                        self.update_goal(goal, goal, low_actions, order)
                        act = self.pick_low_action(obs)
                        success = True
                        break
                if not success:
                    success, goal, bus_goal, low_actions, order = self.generate_goal(True, obs)
                    candidate_goals.append(goal)
                    candidate_orders.append(order)
                    if success:
                        self.update_goal(goal, bus_goal, low_actions, order)
                        act = self.pick_low_action(obs)
        return act

    def pick_low_action(self, obs):
        # Safe and there is no queued low actions, just do nothing
        if self.is_safe(obs) and self.low_len == -1:
            act = self.action_space()
            return act

        # optimize low actions every step
        self.low_actions = self.optimize_low_actions(obs, self.low_actions)
        self.low_len += 1

        # queue has been empty after optimization. just do nothing
        if len(self.low_actions) == 0:
            act = self.action_space()
            self.low_len = -1
        
        # normally execute low action from low actions queue
        else:
            sub_id, new_topo = self.low_actions.pop(0)[:2]
            act = self.converter.convert_act(sub_id, new_topo)
        
        # When it meets maximum low action execution time, log and reset
        if self.max_low_len <= self.low_len:
            self.low_len = -1
        return act

    def high_act(self, stacked_state, adj, sample=True):
        order = None
        with torch.no_grad():
            state = self.emb(stacked_state, adj).detach()
            if sample:
                action = self.actor.sample(state, adj)[0]
            else:
                action = self.actor.mean(state, adj)
        if self.use_order:
            action, order = action
        if order is not None: order = order.detach().cpu()
        return action.detach().cpu(), order
                                                 
    def make_candidate_goal(self, stacked_state, adj, sample, obs):
        goal, order = self.high_act(stacked_state, adj, sample)
        bus_goal = torch.zeros_like(goal).long()
        bus_goal[goal > self.bus_thres] = 1
        low_actions = self.converter.plan_act(bus_goal, obs, order)
        low_actions = self.optimize_low_actions(obs, low_actions)
        return goal, bus_goal, low_actions, order

    def generate_goal(self, sample, obs):
        stacked_state = self.get_current_state().to(self.device)
        adj = self.adj.unsqueeze(0)
        trial = 0
        dn_tried = False
        success = False     # If we found available goal (Not dying act)
        if self.sim_trial == 0:
            goal, bus_goal, low_actions, order = self.make_candidate_goal(stacked_state, adj, sample, obs)
            return success, goal, bus_goal, low_actions, order
        while trial < self.sim_trial:
            # Must generate goal at least once
            goal, bus_goal, low_actions, order = self.make_candidate_goal(stacked_state, adj, sample, obs)
            act = self.action_space() if len(low_actions) == 0 \
                    else self.converter.convert_act(*low_actions[0][:2])
            if act == self.action_space():
                if dn_tried:
                    # 'do nothing' has been already tried and failed, not worthy to try again.
                    trial += 1
                    continue
                else:
                    dn_tried = True
            obs_s = obs.simulate(act)[0]
            # Got available goal!
            if type(obs_s) != type(None):
                success = True
                break
            trial += 1
        return success, goal, bus_goal, low_actions, order
    
    def update_goal(self, goal, bus_goal, low_actions, order=None):
        self.order = order
        self.goal = goal
        self.bus_goal = bus_goal
        self.low_actions = low_actions
        self.low_len = 0

    def optimize_low_actions(self, obs, low_actions):
        # remove overlapped action
        optimized = []
        cooldown_list = obs.time_before_cooldown_sub
        for low_act in low_actions:
            sub_id, sub_goal = low_act[:2]
            sub_goal, same = self.converter.inspect_act(sub_id, sub_goal, obs)
            if not same:
                optimized.append((sub_id, sub_goal, cooldown_list[sub_id]))
        # sort by cooldown_sub
        optimized = sorted(optimized, key=lambda x: x[2])
        
        # if current action has cooldown, then discard
        if len(optimized) > 0 and optimized[0][2] > 0:
            optimized = []
        return optimized
 
            
    def save_model(self, path, name):
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor_%s.pt' % name))
        torch.save(self.emb.state_dict(), os.path.join(path, 'emb_%s.pt' % name))

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pt'), map_location=self.device))
        self.emb.load_state_dict(torch.load(os.path.join(path, 'emb.pt'), map_location=self.device))