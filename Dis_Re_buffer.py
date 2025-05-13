import numpy as np
from copy import deepcopy as dc
import torch
class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, args):
        self.storage = []
        self.max_size = args.capacity
        self.ptr = 0
        self.storage_obs = []
    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, g_x, g_y, u, r, d = [], [], [], [], [],[],[]

        for i in ind:
            X, Y, G_X, G_Y, U, R, D = self.storage[i]
            x.append(dc(X))
            y.append(dc(Y))
            g_x.append(dc(G_X))
            g_y.append(dc(G_Y))
            u.append(dc(U))
            r.append(np.array(dc(R), copy=False))
            d.append(np.array(dc(D), copy=False))

        local_states=self.merge_dicts(x)
        local_states_next = self.merge_dicts(y)
        global_state=self.merge_dicts(g_x)
        global_state_next=self.merge_dicts(g_y)
        dispatching_actions = self.merge_dicts(u)

        return local_states, local_states_next, global_state,global_state_next, dispatching_actions, np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def push_obs(self,data):
        if len(self.storage) == self.max_size:
            self.storage_obs[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage_obs.append(data)

    def merge_dicts(self, dict_list):
        merged_dict = {}

        # 遍历所有字典
        for d in dict_list:
            for key, value in d.items():
                if key not in merged_dict:
                    merged_dict[key] = []
                if isinstance(value, torch.Tensor):
                    merged_dict[key].append(value.detach().clone())
                else:
                    merged_dict[key].append(value)



#        for key in merged_dict:
#            merged_dict[key] = torch.stack(merged_dict[key])
        for key in merged_dict:
            if isinstance(merged_dict[key][0], torch.Tensor):
                merged_dict[key] = torch.stack(merged_dict[key])
            else:
                merged_dict[key] = np.stack(merged_dict[key])
        return merged_dict




