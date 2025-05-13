from ast import arg
from Adv_Re_buffer import Replay_buffer as adv_re_buffer
from Dis_Re_buffer import Replay_buffer as dis_re_buffer
import os
import argparse
import numpy as np
# from online.Reward.rewards import EPRIReward
import torch
import torch.optim as optim
import torch.nn.functional as F
from numpy import *
from Dispatching_Agents.Actor import Dis_actor as dispatch_actor
from Dispatching_Agents.Critic import Dis_critic as dispatch_critic
from Dispatching_Agents.Critic import Hybrid_critic
from Adversarial_Agent.Adver_Agent import adversarial_agent as adv_dqn

from Pretrain_Actor import supervised_train_actors
from Pretrain_Critic import supervised_train_critics

from cmath import inf
from utilize.form_action import *
from Environment.base_env import Environment
from utilize.settings import settings
import csv
import torch.utils.data as Data
import random
import sys
from copy import deepcopy as dc
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

start_idx_set = 42312 #select scenario
HC_hideen_dim=6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parsers():
    parser = argparse.ArgumentParser()

    parser.add_argument('--r_pre', default=0, type=float) # r_pre
    parser.add_argument('--c_loss_pre', default=100, type=float) # c_loss_pre
    parser.add_argument('--b_pre', default=100, type=float)  # b_pre
    

    parser.add_argument('--a_learning_rate', default=1e-7, type=float)#a_learning_rate
    parser.add_argument('--c_learning_rate', default=5e-7, type=float)#c_learning_rate
    parser.add_argument('--b_learning_rate', default=1e-7, type=float)  # c_learning_rate

    parser.add_argument('--exp_name', type=str, default="HCA-MARL_hc6_toge_ELU_42312")
    
    parser.add_argument('--gamma', default=0.95, type=float) # discounted factor
    parser.add_argument('--bre_gamma', default=0.95, type=float) # discounted factor
    parser.add_argument('--tau_1',  default=0.01, type=float) # target smoothing coefficient
    parser.add_argument('--tau_2', default=0.005, type=float)  # target smoothing coefficient

    parser.add_argument('--lam', default=0.03, type=float)  # branch failure probability
    parser.add_argument('--test_iteration', default=10, type=int)
    parser.add_argument('--t_train_gan', default=5, type=int)
    parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
    parser.add_argument('--batch_size', default=64, type=int) # mini batch size
    parser.add_argument('--seed', default=False, type=bool)
    parser.add_argument('--random_seed', default=9527, type=int)
    parser.add_argument('--epsilon', default=0.9, type=float)
    parser.add_argument('--min_epsilon', default=0.05, type=float)
    # optional parameters

    parser.add_argument('--sample_frequency', default=2000, type=int)
    #parser.add_argument('--render', default=False, type=bool) # show UI or not
    parser.add_argument('--log_interval', default=50, type=int) #


    parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
    parser.add_argument('--max_episode', default=200000, type=int)
    parser.add_argument('--print_log', default=5, type=int)
    parser.add_argument('--update_iteration', default=20, type=int)
    parser.add_argument('--pre_training', action="store_true", default=False, help="pre training")
    parser.add_argument('--format', type=int, default=0, help='the format of datas')
    return parser.parse_args()

def get_Global_state(_obs):
    global_state={}
    global_state["branch"] = torch.tensor(np.vstack((_obs.a_ex,_obs.p_ex ,_obs.q_ex , _obs.v_ex , _obs.a_or, _obs.p_or , _obs.q_or , _obs.v_or , _obs.rho)),dtype=torch.float, device=device) #9*185
    global_state["load"]=torch.tensor(np.vstack((_obs.nextstep_load_p , _obs.load_q)),dtype=torch.float, device=device)#2*91
    global_state["else"]=torch.tensor(_obs.grid_loss + _obs.gen_q + _obs.nextstep_renewable_gen_p_max,dtype=torch.float, device=device) #1+54+18=73
    return global_state



def get_Area_states(_obs):
    states_list={}
    for area in settings.area_name:
        state=[]
        for local_branch in getattr(settings,area+"_branch"):
            branch_id=settings.lnname.index(local_branch)
            state=  state+[_obs.a_ex[branch_id]] + [_obs.p_ex[branch_id]] + [_obs.q_ex[branch_id]] + [_obs.v_ex[branch_id]] + \
                    [_obs.a_or[branch_id]] + [_obs.p_or[branch_id]] + [_obs.q_or[branch_id]] + [_obs.v_or[branch_id]] + [_obs.rho[branch_id]]

        for local_gen in getattr(settings,area+"_gen"):
            gen_id=settings.name_index[local_gen]
            state=  state+ [_obs.gen_q[gen_id]]
            if gen_id in settings.renewable_ids:
                state = state + [_obs.nextstep_renewable_gen_p_max[settings.renewable_ids.index(gen_id)]]

        state= state+_obs.grid_loss + _obs.nextstep_load_p + _obs.load_q

        states_list[area]=state

    return states_list


def change_adj_p(obs, actions_p, num_gen):
        adjust_gen_p_action_space = obs.action_space['adjust_gen_p']
        min_adj_p = adjust_gen_p_action_space.low
        max_adj_p = adjust_gen_p_action_space.high
        for i in range(num_gen):
            if min_adj_p[i] == -inf:
                actions_p[i] = 0
                continue
            if min_adj_p[i] == inf:
                actions_p[i] = 0
                continue
            if actions_p[i] < min_adj_p[i]:
                actions_p[i] = min_adj_p[i]
            elif actions_p[i] > max_adj_p[i]:
                actions_p[i] = max_adj_p[i]
        return actions_p

def wrap_action(adjust_gen_p, adjust_gen_v):
        act = {
            'adjust_gen_p': adjust_gen_p,
            'adjust_gen_v': adjust_gen_v
        }
        return act

def change_adj_v(obs, actions_v, num_gen): #最大最小值做了约束
        adjust_gen_v_action_space = obs.action_space['adjust_gen_v']
        min_adj_v = adjust_gen_v_action_space.low
        max_adj_v = adjust_gen_v_action_space.high
        for i in range(num_gen):
            if min_adj_v[i] == -inf or min_adj_v[i] == inf:
                actions_v[i] = 0   #平衡机组动作设为0
                continue
            if actions_v[i] < min_adj_v[i]:
                actions_v[i] = min_adj_v[i]
            elif actions_v[i] > max_adj_v[i]:
                actions_v[i] = max_adj_v[i]
        return actions_v

class Agent():
    def __init__(self, pre_training=True):
   
        self.num_gen = settings.num_gen
        self.rand=True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.epsilon=args.epsilon
        self.area_name_list=settings.area_name
        self.num_area=settings.num_area
        self.Dispatch_agents={}
        all_critic_para=[]
        all_actor_para=[]
        for area in self.area_name_list:
            self.Dispatch_agents[area]={}
            self.Dispatch_agents[area]["state_dim"]= len(getattr(settings,area+"_branch"))*9+len(getattr(settings,area+"_gen"))+len(settings.ldname)*2+1
            for local_gen in getattr(settings, area + "_gen"):
                gen_id = settings.name_index[local_gen]
                if gen_id in settings.renewable_ids:
                    self.Dispatch_agents[area]["state_dim"] = self.Dispatch_agents[area]["state_dim"]+1

            self.Dispatch_agents[area]["action_dim"] = len(getattr(settings,area+"_gen"))

            self.Dispatch_agents[area]["actor"]=dispatch_actor(self.Dispatch_agents[area]["state_dim"],self.Dispatch_agents[area]["action_dim"]).to(self.device)
            self.Dispatch_agents[area]["actor_target"] = dispatch_actor(self.Dispatch_agents[area]["state_dim"],self.Dispatch_agents[area]["action_dim"]).to(self.device)
            all_actor_para = all_actor_para + list(self.Dispatch_agents[area]["actor"].parameters())


            self.Dispatch_agents[area]["critic"] = dispatch_critic(self.Dispatch_agents[area]["state_dim"],self.Dispatch_agents[area]["action_dim"]).to(self.device)
            self.Dispatch_agents[area]["critic_target"] = dispatch_critic(self.Dispatch_agents[area]["state_dim"],self.Dispatch_agents[area]["action_dim"]).to(self.device)
            all_critic_para=all_critic_para+list(self.Dispatch_agents[area]["critic"].parameters())


        self.Dispatch_agents["hybrid_critic"] = Hybrid_critic(self.num_area, HC_hideen_dim).to(self.device)
        all_critic_para = all_critic_para + list(self.Dispatch_agents["hybrid_critic"].parameters())
        self.Dispatch_agents["critic_optimizer"] = torch.optim.Adam(all_critic_para, lr=args.c_learning_rate)
        self.Dispatch_agents["critic_StepLR"] = torch.optim.lr_scheduler.StepLR(self.Dispatch_agents["critic_optimizer"], step_size=100, gamma=0.8)

        self.Dispatch_agents["actor_optimizer"] = optim.Adam(all_actor_para,lr=args.a_learning_rate)
        self.Dispatch_agents["actor_StepLR"] = torch.optim.lr_scheduler.StepLR(
            self.Dispatch_agents["actor_optimizer"], step_size=100, gamma=0.8)


        self.Adver_agent=adv_dqn().to(self.device)
        self.Adver_agent_target = adv_dqn().to(self.device)
        self.Adver_optimizer = optim.Adam(self.Adver_agent.parameters(), lr=args.b_learning_rate)
        self.StepLR_Adver = torch.optim.lr_scheduler.StepLR(self.Adver_optimizer, step_size=100, gamma=0.8)

        self.dis_replay_buffer = dis_re_buffer(args)

        self.bre_replay_buffer = adv_re_buffer(args)

        # self.num_critic_update_iteration = 0
        # self.num_actor_update_iteration = 0
        self.num_dqn_update=0
        
        if pre_training:
            self.is_pre = self.pre_training(self.Dispatch_agents)
        else:
            self.is_pre = False

    def pre_training(self,Dispatch_agents):
        print("Pre training...")
        actor_result = supervised_train_actors(start_idx_set, Dispatch_agents,args.exp_name)

        # critic_result = supervised_train_critic(self.D_critic, format=args.format, exp_name=args.exp_name)
        return True

    def hard_update_target_networks(self):
        for area in self.area_name_list:
            self.Dispatch_agents[area]["actor_target"].load_state_dict(self.Dispatch_agents[area]["actor"].state_dict())
            self.Dispatch_agents[area]["critic_target"].load_state_dict(self.Dispatch_agents[area]["critic"].state_dict())
        self.Adver_agent_target.load_state_dict(self.Adver_agent.state_dict())


    def load(self,episode):
        if self.is_pre:
            print("Load from pre training model...")
            self.is_pre = False
        else:#优先读best.pth

            if os.path.exists(os.path.join("log", args.exp_name, 'best_model_para','hyb_critic_best.pth')):
                print("Load the hyb_critic from the best model...")
                model_path_hc = os.path.join("log", args.exp_name, 'best_model_para', 'hyb_critic_best.pth')
                self.Dispatch_agents["hybrid_critic"].load_state_dict(torch.load(model_path_hc, map_location=self.device))
            else:
                print("Load the hyb_critic from history pretrain model...")
                model_path_hc = os.path.join("log", args.exp_name, 'pretrain_model_para', 'pretrain_hyb_critic.pth')
                self.Dispatch_agents["hybrid_critic"].load_state_dict(torch.load(model_path_hc, map_location=self.device))


            if os.path.exists(os.path.join("log", args.exp_name, 'best_model_para','Adver_dqn_best.pth')):
                print("Load the adver agent from the best model...")
                model_path_ad = os.path.join("log", args.exp_name, 'best_model_para','Adver_dqn_best.pth')
                self.Adver_agent.load_state_dict(torch.load(model_path_ad, map_location=self.device))

            for area in self.area_name_list:
                if not os.path.exists(os.path.join("log",args.exp_name,'best_model_para', area+'_actor_best.pth')):
                    print("Load the actor from the history pretrain model...")
                    model_path_p = os.path.join("log", args.exp_name, 'pretrain_model_para', area+"_pretrain_actor.pth")
                    self.Dispatch_agents[area]["actor"].load_state_dict(torch.load(model_path_p,  map_location=self.device))
                else:
                    print("Load the actor from best model...")
                    model_path_p = os.path.join("log",args.exp_name,'best_model_para', area+'_actor_best.pth')
                    self.Dispatch_agents[area]["actor"].load_state_dict(torch.load(model_path_p,  map_location=self.device))


                if not os.path.exists(os.path.join("log",args.exp_name,'best_model_para', area+'_critic_best.pth')):
                    print("Load the critic from the pretrain history model...")
                    model_path_r = os.path.join("log", args.exp_name, 'pretrain_model_para', area+"_pretrain_critic.pth")
                    self.Dispatch_agents[area]["critic"].load_state_dict(torch.load(model_path_r,  map_location=self.device))

                else:
                    print("Load the critic from best model...")
                    model_path_r = os.path.join("log",args.exp_name,'best_model_para', area+'_critic_best.pth')
                    self.Dispatch_agents[area]["critic"].load_state_dict(torch.load(model_path_r,  map_location=self.device))


                self.Dispatch_agents[area]["actor"] = self.Dispatch_agents[area]["actor"].to(self.device)
                self.Dispatch_agents[area]["actor_target"] = self.Dispatch_agents[area]["actor_target"].to(self.device)
                self.Dispatch_agents[area]["critic"] = self.Dispatch_agents[area]["critic"].to(self.device)
                self.Dispatch_agents[area]["critic_target"] =  self.Dispatch_agents[area]["critic_target"].to(self.device)


        if episode==0:
            self.hard_update_target_networks()


        self.Adver_agent = self.Adver_agent.to(self.device)
        self.Adver_agent_target = self.Adver_agent_target.to(self.device)
        self.Dispatch_agents["hybrid_critic"]=self.Dispatch_agents["hybrid_critic"].to(self.device)

            
    def act(self, obs, global_state=None,local_states=None):
        global_actions,action_p, action_v = self.get_dis_action(obs,local_states) #获得了两个action
        ret_action = wrap_action(action_p, action_v) #都是变化量
        break_line = False
        if np.random.rand() < args.lam:
            break_action=self.get_break_action(global_state)
            break_line=True
            return global_actions, ret_action,break_action,break_line
        else:
            return global_actions,ret_action, None,break_line

    def get_dis_action(self, obs,local_states):

        orig_area_actions={}
        for area in self.area_name_list:
            self.Dispatch_agents[area]["actor"].eval()
            local_action = self.Dispatch_agents[area]["actor"](torch.Tensor([local_states[area]]).to(self.device))
            local_action = local_action.cpu().detach().numpy().tolist()
            local_action = local_action[0]
            orig_area_actions[area]=local_action

        output_p=[]
        for i in range(self.num_gen):
            gen_name=[k for k, v in settings.name_index.items() if v == i][0]
            for area in self.area_name_list:
                if gen_name in getattr(settings,area+"_gen"):
                    gen_local_index=getattr(settings,area+"_gen").index(gen_name)
                    output_p.append(orig_area_actions[area][gen_local_index])
                    break

        action_p_orig = []
        action_v_orig=[]

        for i in range(self.num_gen):#求下一时刻的发电量跟当前时刻发电量差值
            action_p_orig.append(output_p[i] - obs.gen_p[i])
            action_v_orig.append(0) #电压不控制
        action_v = change_adj_v(obs, action_v_orig, self.num_gen)#是否动作在上下限中，

        if self.rand:
            random_list = random.sample(range(0,self.num_gen),15)#加入随机数当作扰动
            for i in range(15):
                action_p_orig[random_list[i]] = action_p_orig[random_list[i]] + random.uniform(-7,7)
                # action_p_orig[random_list[i]] = action_p_orig[random_list[i]]



        action_p = change_adj_p(obs,action_p_orig,self.num_gen)

        area_actions={}
        for area in self.area_name_list:
            area_actions[area]=[]
            for local_gen in getattr(settings,area+"_gen"):
                gen_id=settings.name_index[local_gen]
                area_actions[area].append(action_p[gen_id]+ obs.gen_p[gen_id])

        return area_actions,action_p, action_v

    def get_break_action(self,state_all_repm):
        if np.random.rand() < self.epsilon:
            action_prob = F.softmax(
                torch.tensor(np.random.rand(1, settings.num_attackable_branch), device=self.device, dtype=torch.float32),
                dim=1)
        else:
            with torch.no_grad():
                self.Adver_agent.eval()
                action_prob = self.Adver_agent(state_all_repm)
        action = torch.argmax(action_prob, dim=1).item()
        return action


    def train_dis_agent(self,time_a,time_c):
        # all_actor_loss=[]
        # all_critic_loss = []


        # Sample replay buffer
        local_states, local_states_next, global_state,global_state_next, dispatching_actions, r, d = self.dis_replay_buffer.sample(args.batch_size)
        done = torch.FloatTensor(1 - d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)

        target_local_q_list=[]
        current_local_q_list=[]
        for area in self.area_name_list:
            self.Dispatch_agents[area]["actor_target"].eval()
            self.Dispatch_agents[area]["critic_target"].eval()
            local_state = torch.FloatTensor(local_states[area]).to(self.device)
            next_local_state = torch.FloatTensor(local_states_next[area]).to(self.device)
            local_action = torch.FloatTensor(dispatching_actions[area]).to(self.device)

        # Compute the local target Q value
            target_local_Q = self.Dispatch_agents[area]["critic_target"](next_local_state,
                                            self.Dispatch_agents[area]["actor_target"](next_local_state))
            target_local_q_list.append(target_local_Q)

        # Get current local Q estimate
            current_Q = self.Dispatch_agents[area]["critic"](local_state, local_action)
            current_local_q_list.append(current_Q)


        taeget_qs_tensor = torch.cat(target_local_q_list, dim=1).view(-1, 1, settings.num_area)
        current_qs_tensor = torch.cat(current_local_q_list, dim=1).view(-1, 1, settings.num_area)

        total_target_q = self.Dispatch_agents["hybrid_critic"](taeget_qs_tensor, global_state_next)
        total_current_q = self.Dispatch_agents["hybrid_critic"](current_qs_tensor, global_state)

        target_Q = reward + (done * args.gamma * total_target_q).detach()
        # Compute critic loss
        critic_loss = F.mse_loss(total_current_q, target_Q)
        # all_critic_loss.append(critic_loss.item())
        #self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
        # Optimize the critic
        self.Dispatch_agents["critic_optimizer"].zero_grad()
        critic_loss.backward()
        self.Dispatch_agents["critic_optimizer"].step()
        
            # print("current_lr: %s" % (self.critic_optimizer.state_dict()['param_groups'][0]['lr']))

        action_local_q_list=[]
        for area in self.area_name_list:
        # Compute actor loss
            local_state = torch.FloatTensor(local_states[area]).to(self.device)
            local_action = self.Dispatch_agents[area]["actor"](local_state)

            action_local_q = self.Dispatch_agents[area]["critic"](local_state,local_action)
            action_local_q_list.append(action_local_q)

        action_qs_tensor = torch.cat(action_local_q_list, dim=1).view(-1, 1, settings.num_area)
        actor_loss = -self.Dispatch_agents["hybrid_critic"](action_qs_tensor, global_state).mean()
        #self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
        # all_actor_loss.append(actor_loss.item())
        # Optimize the actor
        self.Dispatch_agents["actor_optimizer"].zero_grad()
        actor_loss.backward()
        self.Dispatch_agents["actor_optimizer"].step()

        # Update the frozen target models
        if time_c % 5 ==0:
            for area in self.area_name_list:
                for param, target_param in zip(self.Dispatch_agents[area]["critic"].parameters(), self.Dispatch_agents[area]["critic_target"].parameters()):
                    target_param.data.copy_(args.tau_2 * param.data + (1 - args.tau_2) * target_param.data)
        if time_a % 5 ==0:
            for area in self.area_name_list:
                for param, target_param in zip(self.Dispatch_agents[area]["actor"].parameters(), self.Dispatch_agents[area]["actor_target"].parameters()):
                        target_param.data.copy_(args.tau_2 * param.data + (1 - args.tau_2) * target_param.data)

        self.epsilon=max(args.min_epsilon,self.epsilon-0.0001*time_a)


        return critic_loss.item(), actor_loss.item()

    def train_break_agent(self,time_b):
        self.Adver_agent_target.eval()
        #self.Adver_agent.eval()

        x, y, u, r, d = self.bre_replay_buffer.sample(args.batch_size)

        state = x
        action = torch.tensor(u).view(-1,1).to(self.device)
        next_state = y
        done = torch.FloatTensor(1 - d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)

        Q_values = self.Adver_agent(state).gather(1,action)

        next_Q_values = self.Adver_agent_target(next_state)
        # 下一个状态每一个Q值
        next_Q_values = next_Q_values.max(1)[0].view(-1, 1)

        target_Q_values = reward + args.bre_gamma * (next_Q_values * done)

        self.Adver_optimizer.zero_grad()
        loss = F.mse_loss(Q_values, target_Q_values)  # mse loss函数
        loss.backward()

        self.Adver_optimizer.step()

        if time_b % 5 ==0:
            for param, target_param in zip(self.Adver_agent.parameters(), self.Adver_agent_target.parameters()):
                target_param.data.copy_(args.tau_1 * param.data + (1 - args.tau_1) * target_param.data)

        return loss

    def save_dispatch_actors(self):
        if not os.path.exists(os.path.join("log", args.exp_name, 'best_model_para')):
            os.makedirs(os.path.join("log", args.exp_name, 'best_model_para'))
        print("Save the actor network...")
        for area in self.area_name_list:
            save_path = os.path.join("log", args.exp_name, 'best_model_para', area+'_actor_best.pth')
            torch.save(self.Dispatch_agents[area]["actor"].state_dict(), save_path)
        self.Dispatch_agents["actor_StepLR"].step()

    def save_dispatch_ctitics(self):
        if not os.path.exists(os.path.join("log", args.exp_name, 'best_model_para')):
            os.makedirs(os.path.join("log", args.exp_name, 'best_model_para'))
        print("Save the critic network...")

        model_path_hc = os.path.join("log", args.exp_name, 'best_model_para', 'hyb_critic_best.pth')
        torch.save(self.Dispatch_agents["hybrid_critic"].state_dict(), model_path_hc)

        for area in self.area_name_list:
            save_path = os.path.join("log", args.exp_name, 'best_model_para', area+'_critic_best.pth')
            torch.save(self.Dispatch_agents[area]["critic"].state_dict(), save_path)
        self.Dispatch_agents["critic_StepLR"].step()

    def save_adver_agent(self):
        if not os.path.exists(os.path.join("log", args.exp_name, 'best_model_para')):
            os.makedirs(os.path.join("log", args.exp_name, 'best_model_para'))
        print("Save the break_dqn network...")
        save_path = os.path.join("log", args.exp_name, 'best_model_para','Adver_dqn_best.pth')
        torch.save(self.Adver_agent.state_dict(), save_path)
        (self.StepLR_Adver.step())


    def test_result(self):
        total_reward = 0
        step = 0
        env = Environment(settings, "EPRIReward")
        obs, start_idx = env.reset(start_sample_idx=int(start_idx_set))

        
        global_state = get_Global_state(obs)
        local_state=get_Area_states(obs)
        test_total_reward_t_lists = {"EPRIReward":[], "line_over_flow_reward":[], "renewable_consumption_reward":[] ,
                            "running_cost_reward":[] ,"balanced_gen_reward":[] ,"gen_reactive_power_reward":[] ,"sub_voltage_reward":[],"running_cost":[]}
        while True:

            area_actions,dis_action,break_action,if_break_line = self.act(obs, global_state,local_state)


            print("test****************",step)
            #print(action_p)
            next_obs, reward_lists, done, info = env.step(dis_action,break_action,if_break_line)
            # print("List:")
            # print(reward_lists)
            if reward_lists == 0:
                reward = 0
            else:
                reward = reward_lists["EPRIReward"]
            total_reward += reward
            if reward != 0:
                test_total_reward_t_lists["EPRIReward"].append(reward_lists["EPRIReward"])
                test_total_reward_t_lists["line_over_flow_reward"].append(reward_lists["line_over_flow_reward"])
                test_total_reward_t_lists["renewable_consumption_reward"].append(reward_lists["renewable_consumption_reward"])
                test_total_reward_t_lists["running_cost_reward"].append(reward_lists["running_cost_reward"])
                test_total_reward_t_lists["balanced_gen_reward"].append(reward_lists["balanced_gen_reward"])
                test_total_reward_t_lists["gen_reactive_power_reward"].append(reward_lists["gen_reactive_power_reward"])
                test_total_reward_t_lists["sub_voltage_reward"].append(reward_lists["sub_voltage_reward"])
                test_total_reward_t_lists["running_cost"].append( reward_lists["running_cost"])
            print("step_reward:",reward)
            #r_pre=total_reward
            if abs(reward-0)<0.00001:   #给更大的惩罚
                reward=-2
        
            global_state_next = get_Global_state(next_obs)#下一时刻
            local_state_next = get_Area_states(next_obs)

            #if args.render and i >= args.render_interval: env.render()
#            if reward>0.5 or len(self.dis_replay_buffer.storage) < 500*settings.num_area:
            if True:
                self.dis_replay_buffer.push((dc(local_state), dc(local_state_next), dc(global_state), dc(global_state_next), dc(area_actions), dc(reward), dc(np.float(done))))
#            if (reward<1 or len(self.bre_replay_buffer.storage) < 500) and if_break_line:
            if if_break_line:
                self.bre_replay_buffer.push((dc(global_state), dc(global_state_next), dc(break_action), dc(-reward), dc(np.float(done))))

            global_state = dc(global_state_next)
            obs = dc(next_obs)
            local_state=dc(local_state_next)
            step += 1
            if done:
                break

        return total_reward, step, test_total_reward_t_lists

if __name__ == '__main__':
    args = get_parsers()
    save_Test=False
    if not os.path.exists(os.path.join("log", args.exp_name)):
        os.makedirs(os.path.join("log", args.exp_name))
        print("make the new dirs...")
    pre_tra = False
    agent=Agent(pre_training=pre_tra)
    # total_step = 0
    r_pre = args.r_pre
    c_loss_pre = args.c_loss_pre #critic
    b_pre=args.b_pre

    time_a = 0 #update times
    time_c = 0
    time_b = 0
    if True:
        with open(os.path.join('log', args.exp_name, "result_online_best_reward.csv"),"a+",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["episode","train_times","update_times","steps","EPRIReward","line_over_flow_reward","renewable_consumption_rewards","running_cost_reward","balanced_gen_reward","gen_reactive_power_reward","sub_voltage_reward","total_cost","A loss","C loss"])

        with open(os.path.join('log', args.exp_name, "result_online_loss.csv"),"a+",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["episode","train_times","update_times","steps","total reward","A loss","C loss"])

        with open(os.path.join('log', args.exp_name, "result_best_epoch_reward.csv"),"a+",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["step","EPRIReward","line_over_flow_reward","renewable_consumption_rewards","running_cost_reward","balanced_gen_reward","gen_reactive_power_reward","sub_voltage_reward","cost"])

        with open(os.path.join('log', args.exp_name, "test_reward.csv"),"a+",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["episode","update_times","steps","EPRIReward","line_over_flow_reward","renewable_consumption_rewards","running_cost_reward","balanced_gen_reward","gen_reactive_power_reward","sub_voltage_reward","cost"])

        with open(os.path.join('log', args.exp_name, "result_loss.csv"),"a+",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["episode","Average a loss","Average c loss"])

    for i in range(args.max_episode):
        # total_reward = 0
        # step = 0
        agent.load(i)


        total_reward,step,test_total_reward_t_lists=agent.test_result()
        print("test_total_reward:",total_reward)#测试结束
        # total_step += step + 1
        if r_pre>=50:
            agent.rand=True
        if save_Test or total_reward>r_pre:#
            with open(os.path.join('log', args.exp_name, "test_reward.csv"),"a+",newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([ i,time_a,step,
                                          sum(test_total_reward_t_lists["EPRIReward"]),
                                          sum(test_total_reward_t_lists["line_over_flow_reward"]),
                                          sum(test_total_reward_t_lists["renewable_consumption_reward"]),
                                          sum(test_total_reward_t_lists["running_cost_reward"]),#除了cost
                                          sum(test_total_reward_t_lists["balanced_gen_reward"]),#
                                          sum(test_total_reward_t_lists["gen_reactive_power_reward"]),#
                                          sum(test_total_reward_t_lists["sub_voltage_reward"]),#
                                          sum(test_total_reward_t_lists["running_cost"])])#除了cost
            save_Test=False
        if total_reward>r_pre*0.9:
            r_pre=max(total_reward,r_pre)
            agent.save_dispatch_actors()
            with open(os.path.join('log', args.exp_name, "result_online_best_reward.csv"),"a+",newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([ i, 0, time_a,step,
                                          sum(test_total_reward_t_lists["EPRIReward"]),
                                          sum(test_total_reward_t_lists["line_over_flow_reward"]),
                                          sum(test_total_reward_t_lists["renewable_consumption_reward"]),
                                          sum(test_total_reward_t_lists["running_cost_reward"]),
                                          sum(test_total_reward_t_lists["balanced_gen_reward"]),
                                          sum(test_total_reward_t_lists["gen_reactive_power_reward"]),
                                          sum(test_total_reward_t_lists["sub_voltage_reward"]),
                                          sum(test_total_reward_t_lists["running_cost"]),
                                          0, 0])
            with open(os.path.join('log', args.exp_name, "result_best_epoch_reward.csv"), "a+",
                      newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Step", "EPRIReward", "line_over_flow_reward", "renewable_consumption_rewards",
                                 "running_cost_reward", "balanced_gen_reward", "gen_reactive_power_reward",
                                 "sub_voltage_reward","cost"])
                for step_num in range(len(test_total_reward_t_lists["EPRIReward"])):
                    writer.writerow([step_num, test_total_reward_t_lists["EPRIReward"][step_num],
                                     test_total_reward_t_lists["line_over_flow_reward"][step_num],
                                     test_total_reward_t_lists["renewable_consumption_reward"][step_num],
                                     test_total_reward_t_lists["running_cost_reward"][step_num],
                                     test_total_reward_t_lists["balanced_gen_reward"][step_num],
                                     test_total_reward_t_lists["gen_reactive_power_reward"][step_num],
                                     test_total_reward_t_lists["sub_voltage_reward"][step_num],
                                     test_total_reward_t_lists["running_cost"][step_num]])
        
        if len(agent.dis_replay_buffer.storage) > args.batch_size and len(agent.bre_replay_buffer.storage) > args.batch_size:
            total_c_loss = 0 #critic loss
            total_a_loss = 0 #actor loss
            total_b_loss = 0  # actor loss
            iterations = 0
            for train_times in range(args.update_iteration):
                print("****************",train_times,"****************")
                c_loss, a_loss = agent.train_dis_agent(time_a,time_c)
                b_loss=agent.train_break_agent(time_b)
                total_c_loss += c_loss
                total_a_loss += a_loss
                total_b_loss +=b_loss
                iterations += 1

                total_reward_test, step_test, reward_lists = agent.test_result()

                if total_reward_test > r_pre *0.9:
                    agent.save_dispatch_actors()
                    #i, train_times, time_a, total_reward_test
                    r_pre = max(total_reward_test,r_pre)
                    time_a = time_a + 1
                    save_Test = True
                    with open(os.path.join('log', args.exp_name, "result_online_best_reward.csv"),"a+",newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([i, train_times, time_a, step_test,
                                         sum(reward_lists["EPRIReward"]),
                                         sum(reward_lists["line_over_flow_reward"]),
                                         sum(reward_lists["renewable_consumption_reward"]),
                                         sum(reward_lists["running_cost_reward"]),
                                         sum(reward_lists["balanced_gen_reward"]),
                                         sum(reward_lists["gen_reactive_power_reward"]),
                                         sum(reward_lists["sub_voltage_reward"]),
                                         sum(reward_lists["running_cost"])]+
                                         [a_loss] +[c_loss])

                    if total_reward_test > 0: 
                        with open(os.path.join('log', args.exp_name, "result_best_epoch_reward.csv"), "a+", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["Step", "EPRIReward","line_over_flow_reward","renewable_consumption_rewards","running_cost_reward","balanced_gen_reward","gen_reactive_power_reward","sub_voltage_reward","cost"])
                            for step_num in range(len(reward_lists["EPRIReward"])):
                                writer.writerow([step_num,
                                                 reward_lists["EPRIReward"][step_num],
                                                 reward_lists["line_over_flow_reward"][step_num],
                                                 reward_lists["renewable_consumption_reward"][step_num],
                                                 reward_lists["running_cost_reward"][step_num],
                                                 reward_lists["balanced_gen_reward"][step_num],
                                                 reward_lists["gen_reactive_power_reward"][step_num],
                                                 reward_lists["sub_voltage_reward"][step_num],
                                                 reward_lists["running_cost"][step_num]])



                if c_loss < c_loss_pre*1.5:
                    agent.save_dispatch_ctitics()
                        # i, train_times, time_c, a_loss, c_loss
                    c_loss_pre = c_loss
                    time_c = time_c + 1
                    with open(os.path.join('log', args.exp_name, "result_online_loss.csv"), "a+", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([i, train_times, time_c,step_test, total_reward_test]+ [a_loss] +[c_loss]+[b_loss])
                
                print("train_EXP_NAME: {} Episode: \t{} steps: \t{} Total Reward: \t{:0.2f} A_loss: \t{:0.2f} C_loss: \t{:0.2f} R_PRE: \t{:0.2f} Test_reward: \t{:0.2f}".format(args.exp_name, i, step_test,
                                                                                            total_reward_test,a_loss,c_loss,r_pre,total_reward_test))


                if total_reward_test < b_pre:
                    agent.save_adver_agent()
                    time_b = time_b + 1
                    b_pre = total_reward_test

                if step_test<60 and r_pre > 250:
                    break

            with open(os.path.join('log', args.exp_name, "result_loss.csv"), "a+", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([i, total_a_loss/iterations, total_c_loss/iterations,total_b_loss/iterations])
        else:
            print("EXP_NAME: {} Episode: \t{} steps: \t{} Total Reward: \t{:0.2f} R_PRE: \t{:0.2f}".format(args.exp_name, i, step,
                                                                                        total_reward,r_pre))