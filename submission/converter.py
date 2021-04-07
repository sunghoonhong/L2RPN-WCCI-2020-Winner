import numpy as np
import torch
from grid2op.Converter.Converters import Converter


class graphGoalConverter:
    def __init__(self, env, mask, mask_hi, danger, device):
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        self.mask = mask
        self.mask_hi = mask_hi
        self.danger = danger
        self.device = device
        self.thermal_limit_under400 = torch.from_numpy(env._thermal_limit_a < 400)
        self.init_obs_converter()
        self.init_action_converter()

    def init_obs_converter(self):        
        self.idx = self.obs_space.shape
        self.pp = np.arange(sum(self.idx[:6]),sum(self.idx[:7]))
        self.lp = np.arange(sum(self.idx[:9]),sum(self.idx[:10]))
        self.op = np.arange(sum(self.idx[:12]),sum(self.idx[:13]))
        self.ep = np.arange(sum(self.idx[:16]),sum(self.idx[:17]))
        self.rho = np.arange(sum(self.idx[:20]),sum(self.idx[:21]))
        self.topo = np.arange(sum(self.idx[:23]),sum(self.idx[:24]))
        self.main = np.arange(sum(self.idx[:26]),sum(self.idx[:27]))
        self.over = np.arange(sum(self.idx[:22]),sum(self.idx[:23]))
        
        # parse substation info
        self.subs = [{'e':[], 'o':[], 'g':[], 'l':[]} for _ in range(self.action_space.n_sub)]
        for gen_id, sub_id in enumerate(self.action_space.gen_to_subid):
            self.subs[sub_id]['g'].append(gen_id)
        for load_id, sub_id in enumerate(self.action_space.load_to_subid):
            self.subs[sub_id]['l'].append(load_id)
        for or_id, sub_id in enumerate(self.action_space.line_or_to_subid):
            self.subs[sub_id]['o'].append(or_id)
        for ex_id, sub_id in enumerate(self.action_space.line_ex_to_subid):
            self.subs[sub_id]['e'].append(ex_id)
        
        self.sub_to_topos = []  # [0]: [0, 1, 2], [1]: [3, 4, 5, 6, 7, 8]
        for sub_info in self.subs:
            a = []
            for i in sub_info['e']:
                a.append(self.action_space.line_ex_pos_topo_vect[i])
            for i in sub_info['o']:
                a.append(self.action_space.line_or_pos_topo_vect[i])
            for i in sub_info['g']:
                a.append(self.action_space.gen_pos_topo_vect[i])
            for i in sub_info['l']:
                a.append(self.action_space.load_pos_topo_vect[i])
            self.sub_to_topos.append(torch.LongTensor(a))

        # split topology over sub_id
        self.sub_to_topo_begin, self.sub_to_topo_end = [], []
        idx = 0
        for num_topo in self.action_space.sub_info:
            self.sub_to_topo_begin.append(idx)
            idx += num_topo
            self.sub_to_topo_end.append(idx)
        self.max_n_line = max([len(topo['o'] + topo['e']) for topo in self.subs])
        self.max_n_or = max([len(topo['o']) for topo in self.subs])
        self.max_n_ex = max([len(topo['e']) for topo in self.subs])
        self.max_n_g = max([len(topo['g']) for topo in self.subs])
        self.max_n_l = max([len(topo['l']) for topo in self.subs])
        self.n_feature = 6

    def convert_obs(self, o):
        length = self.action_space.dim_topo # N
        p_ = torch.zeros(o.size(0), length).to(o.device)    # (B, N)
        p_[..., self.action_space.gen_pos_topo_vect] = o[...,  self.pp]
        p_[..., self.action_space.load_pos_topo_vect] = o[..., self.lp]
        p_[..., self.action_space.line_or_pos_topo_vect] = o[..., self.op]
        p_[..., self.action_space.line_ex_pos_topo_vect] = o[..., self.ep]

        rho_ = torch.zeros(o.size(0), length).to(o.device)
        rho_[..., self.action_space.line_or_pos_topo_vect] = o[..., self.rho]
        rho_[..., self.action_space.line_ex_pos_topo_vect] = o[..., self.rho]

        danger_ = torch.zeros(o.size(0), length).to(o.device)
        danger = ((o[...,self.rho] >= self.danger-0.05) & self.thermal_limit_under400.to(o.device)) | (o[...,self.rho] >= self.danger)
        danger_[..., self.action_space.line_or_pos_topo_vect] = danger.float()
        danger_[..., self.action_space.line_ex_pos_topo_vect] = danger.float()      

        over_ = torch.zeros(o.size(0), length).to(o.device)
        over_[..., self.action_space.line_or_pos_topo_vect] = o[..., self.over]/3
        over_[..., self.action_space.line_ex_pos_topo_vect] = o[..., self.over]/3

        main_ = torch.zeros(o.size(0), length).to(o.device)
        temp = torch.zeros_like(o[..., self.main])
        temp[o[..., self.main] ==0] = 1
        main_[..., self.action_space.line_or_pos_topo_vect] = temp
        main_[..., self.action_space.line_ex_pos_topo_vect] = temp

        topo_ = o[..., self.topo]
        state = torch.stack([p_, rho_, danger_, topo_, over_, main_], dim=2) # B, N, F
        return state
  
    def init_action_converter(self):
        self.sorted_sub = list(range(self.action_space.n_sub))
        self.sub_mask = []  # mask for parsing actionable topology
        self.psubs = []     # actionable substation IDs
        self.masked_sub_to_topo_begin = []
        self.masked_sub_to_topo_end = []
        idx = 0
        for i, num_topo in enumerate(self.action_space.sub_info):
            if num_topo > self.mask and num_topo < self.mask_hi:
                self.sub_mask.extend(
                    [j for j in range(self.sub_to_topo_begin[i]+1, self.sub_to_topo_end[i])])
                self.psubs.append(i)
                self.masked_sub_to_topo_begin.append(idx)
                idx += num_topo-1
                self.masked_sub_to_topo_end.append(idx)

            else: # dummy
                self.masked_sub_to_topo_begin.append(-1)
                self.masked_sub_to_topo_end.append(-1)
        self.n = len(self.sub_mask)

        self.masked_sorted_sub = [i for i in self.sorted_sub if i in self.psubs]
        self.lonely_lines = set()
        for i in range(self.obs_space.n_line):
            if (self.obs_space.line_or_to_subid[i] not in self.psubs) \
               and (self.obs_space.line_ex_to_subid[i] not in self.psubs):
                self.lonely_lines.add(i)
        self.lonely_lines = list(self.lonely_lines) 

    def inspect_act(self, sub_id, goal, obs):
        # Correct illegal action
        # collect original ids
        exs = self.subs[sub_id]['e']
        ors = self.subs[sub_id]['o']
        lines = exs + ors   # [line_id0, line_id1, line_id2, ...]
        
        # Just prevent isolation
        line_idx = len(lines) - 1 
        if (goal[:line_idx] == 1).all() * (goal[line_idx:] != 1).any():
            goal = torch.ones_like(goal)
        
        if torch.is_tensor(goal): goal = goal.numpy()
        beg = self.masked_sub_to_topo_begin[sub_id]
        end = self.masked_sub_to_topo_end[sub_id]
        already_same = np.all(goal == obs.topo_vect[self.sub_mask][beg:end])
        return goal, already_same

    def convert_act(self, sub_id, new_topo):
        new_topo = [1] + new_topo.tolist()
        act = self.action_space({'set_bus': {'substations_id': [(sub_id, new_topo)]}})
        return act

    def plan_act(self, goal, obs, sub_order_score=None):
        # Compare obs.topo_vect and goal
        # Then parse partial order from whole topological sort
        topo_vect = torch.LongTensor(obs.topo_vect)[self.sub_mask]
        targets = []
        goal = goal.squeeze(0).cpu() + 1
        
        if sub_order_score is None:
            sub_order = self.masked_sorted_sub
        else:
            sub_order = [i[0] for i in sorted(list(zip(self.masked_sorted_sub, sub_order_score[0].tolist())),
                        key=lambda x: -x[1])]
        for sub_id in sub_order:
            beg = self.masked_sub_to_topo_begin[sub_id]
            end = self.masked_sub_to_topo_end[sub_id]
            topo = topo_vect[beg:end]
            new_topo = goal[beg:end]
            if torch.any(new_topo != topo).item():
                targets.append((sub_id, new_topo))
        # plan action sequence from the goal
        plan = [(sub_id, new_topo) for sub_id, new_topo in targets]
        return plan
