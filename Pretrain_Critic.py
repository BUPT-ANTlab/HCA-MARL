import csv
import random
import torch.utils.data as Data
import numpy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import  OrderedDict
from torch import optim
from torch.autograd import Variable

from utilize.settings import settings
from Dispatching_Agents.Critic import Dis_critic
from Dispatching_Agents.Critic import Hybrid_critic
import os
import sys
from copy import deepcopy as dc


start_idx_set = 42312
BATCH_SIZE=512
LR = 0.0001
epoch_num = 100000000000000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str_list_to_float_list(str_list):
    n = 0
    while n < len(str_list):
        str_list[n] = float(str_list[n])
        n += 1
    return(str_list)

def merge_dicts(dict_list):
    merged_dict = {}

    # 遍历所有字典
    for d in dict_list:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = []

            merged_dict[key].append(value)


    for key in merged_dict:
        merged_dict[key] = np.stack(merged_dict[key])
    return merged_dict

def dataset(start_idx_set,train_data_len=800,test_data_len=600):

    pretrain_start = start_idx_set-100
    train_data_end = pretrain_start+train_data_len#分割数据集
    #

    # pretrain_start = 0
    # train_data_end = 100000  # 分割数据集

    test_start = start_idx_set + 1
    test_data_end = test_start + test_data_len

    train_data_state = {}
    train_data_global_state=[]
    test_data_state={}
    test_data_global_state = []
    train_data_action = {}
    test_data_action={}
    train_target = {}
    test_target={}

    train_data_a_ex =[]
    train_data_p_ex =[]
    train_data_v_ex =[]
    train_data_q_ex =[]
    train_data_a_or = []
    train_data_p_or=[]
    train_data_v_or = []
    train_data_q_or = []
    train_data_rho =[]
    train_data_grid_loss =[]
    train_data_load_p =[]
    train_data_load_q =[]
    train_data_max_p =[]
    train_data_gen_q =[]
    train_data_gen_p = []
    train_data_reweard=[]

    test_data_a_ex =[]
    test_data_p_ex =[]
    test_data_v_ex =[]
    test_data_q_ex =[]
    test_data_a_or = []
    test_data_p_or = []
    test_data_v_or = []
    test_data_q_or = []
    test_data_rho =[]
    test_data_grid_loss =[]
    test_data_load_p =[]
    test_data_load_q =[]
    test_data_max_p =[]
    test_data_gen_q =[]
    test_data_gen_p = []
    test_data_reweard =[]

    with open("data/a_ex.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_a_ex.append(row)
            if i>test_start and i<=test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_a_ex.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/p_ex.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_p_ex.append(row)
            if i > test_start and i <= test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_p_ex.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/v_ex.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_v_ex.append(row)
            if i > test_start and i <= test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_v_ex.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/q_ex.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_q_ex.append(row)
            if i > test_start and i <= test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_q_ex.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/a_or.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_a_or.append(row)
            if i > test_start and i <= test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_a_or.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/p_or.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_p_or.append(row)
            if i > test_start and i <= test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_p_or.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/v_or.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_v_or.append(row)
            if i > test_start and i <= test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_v_or.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/q_or.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_q_or.append(row)
            if i > test_start and i <= test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_q_or.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/rho.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_rho.append(row)
            if i>test_start and i<=test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_rho.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/grid_loss.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_grid_loss.append(row)
            if i>test_start and i<=test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_grid_loss.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/load_p.csv", "r", newline="") as csvfile1:
        reader1 = csv.reader(csvfile1)
        for i, rows in enumerate(reader1):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_load_p.append(row)
            if i>test_start and i<=test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_load_p.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/load_q.csv", "r", newline="") as csvfile2:
        reader2 = csv.reader(csvfile2)
        for i, rows in enumerate(reader2):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_load_q.append(row)
            if i>test_start and i<=test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_load_q.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/max_renewable_gen_p.csv", "r", newline="") as csvfile3:
        reader3 = csv.reader(csvfile3)
        for i, rows in enumerate(reader3):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_max_p.append(row)
            if i>test_start and i<=test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_max_p.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/gen_q.csv", "r", newline="") as csvfile4:
        reader4 = csv.reader(csvfile4)
        for i, rows in enumerate(reader4):
            if i > pretrain_start and i <= train_data_end:
                row = rows
                row = str_list_to_float_list(row)
                train_data_gen_q.append(row)
            if i>test_start and i<=test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_gen_q.append(row)
            if i>test_data_end and i>train_data_end:
                break

    with open("data/gen_p.csv", "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
            if i <= train_data_end and i > pretrain_start:
                row = rows
                row = str_list_to_float_list(row)
                train_data_gen_p.append(row)
            if i > test_start and i <= test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_gen_p.append(row)
            if i > test_data_end and i > train_data_end:
                break

    with open("data/reward.csv", "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
            if i <= train_data_end and i > pretrain_start:
                row = rows
                row = str_list_to_float_list(row)
                train_data_reweard.append(row)
            if i > test_start and i <= test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_reweard.append(row)
            if i > test_data_end and i > train_data_end:
                break



    for area in settings.area_name:
        train_area_states = []
        train_area_actions = []
        for i in range(0, len(train_data_gen_q)):
            train_global_state={}
            if area == settings.area_name[0]:
                train_global_state["branch"]=np.vstack((train_data_a_ex[i],train_data_p_ex[i],
                        train_data_q_ex[i] , train_data_v_ex[i],
                        train_data_a_or[i], train_data_p_or[i] ,train_data_q_or[i] , train_data_v_or[i] , train_data_rho[i]))
                train_global_state["load"]=np.vstack((train_data_load_p[i], train_data_load_q[i]))  # 2*91
                train_global_state["else"]=train_data_grid_loss[i] + train_data_gen_q[i] + train_data_max_p[i]
                train_data_global_state.append(train_global_state)
            state = []
            for local_branch in getattr(settings, area + "_branch"):
                branch_id = settings.lnname.index(local_branch)
                state = (state + [train_data_a_ex[i][branch_id]] + [train_data_p_ex[i][branch_id]] +
                         [train_data_q_ex[i][branch_id]] + [train_data_v_ex[i][branch_id]] +
                        [train_data_a_or[i][branch_id]] + [train_data_p_or[i][branch_id]] +
                        [train_data_q_or[i][branch_id]] + [train_data_v_or[i][branch_id]] +
                        [train_data_rho[i][branch_id]])
            for local_gen in getattr(settings, area + "_gen"):
                gen_id = settings.name_index[local_gen]
                state = state + [train_data_gen_q[i][gen_id]]
                if gen_id in settings.renewable_ids:
                    state = state + [train_data_max_p[i][settings.renewable_ids.index(gen_id)]]

            state = state + train_data_grid_loss[i] + train_data_load_p[i] + train_data_load_q[i]
            train_area_states.append(state)

            train_area_action = []
            for local_gen in getattr(settings, area + "_gen"):
                gen_id = settings.name_index[local_gen]
                train_area_action.append(train_data_gen_p[i][gen_id])
            train_area_actions.append(train_area_action)
        # print("***train_data_global_state[branch]", area, ":", np.array(train_area_states).shape,
        #       np.array(train_area_states[0]).shape)
        train_data_state[area] = train_area_states
        # print("***train_data_action", area, ":", np.array(train_area_actions).shape,
        #       np.array(train_area_actions[0]).shape)
        train_data_action[area] = train_area_actions
        # print("***train_data_target", area, ":", np.array(train_data_reweard).shape,
        #       np.array(train_data_reweard[0]).shape)
        train_target[area]= train_data_reweard

        test_area_states = []
        test_area_actions = []
        for i in range(0, len(test_data_gen_q)):  #
            test_global_state = {}
            if area == settings.area_name[0]:
                test_global_state["branch"]=np.vstack((test_data_a_ex[i], test_data_p_ex[i],
                                                                    test_data_q_ex[i], test_data_v_ex[i],
                                                                    test_data_a_or[i], test_data_p_or[i],
                                                                    test_data_q_or[i], test_data_v_or[i],
                                                                    test_data_rho[i]))
                test_global_state["load"]=np.vstack((test_data_load_p[i], test_data_load_q[i]))  # 2*91
                test_global_state["else"]=test_data_grid_loss[i] + test_data_gen_q[i] + test_data_max_p[i]
                test_data_global_state.append(test_global_state)
            state = []
            for local_branch in getattr(settings, area + "_branch"):
                branch_id = settings.lnname.index(local_branch)
                state = state + [test_data_a_ex[i][branch_id]] + [test_data_p_ex[i][branch_id]] + [
                    test_data_q_ex[i][branch_id]] + \
                        [test_data_v_ex[i][branch_id]] + [test_data_a_or[i][branch_id]] + [
                            test_data_p_or[i][branch_id]] + \
                        [test_data_q_or[i][branch_id]] + [test_data_v_or[i][branch_id]] + [test_data_rho[i][branch_id]]
            for local_gen in getattr(settings, area + "_gen"):
                gen_id = settings.name_index[local_gen]
                state = state + [test_data_gen_q[i][gen_id]]
                if gen_id in settings.renewable_ids:
                    state = state + [test_data_max_p[i][settings.renewable_ids.index(gen_id)]]

            state = state + test_data_grid_loss[i] + test_data_load_p[i] + test_data_load_q[i]
            test_area_states.append(state)

            test_area_action = []
            for local_gen in getattr(settings, area + "_gen"):
                gen_id = settings.name_index[local_gen]
                test_area_action.append(test_data_gen_p[i][gen_id])
            test_area_actions.append(test_area_action)
        # print("***test_data_state", area, ":", np.array(test_area_states).shape, np.array(test_area_states[0]).shape)
        test_data_state[area] = test_area_states
        test_data_action[area] = test_area_actions
        test_target[area] = test_data_reweard
        # print("***test_data_state", area, ":", np.array(test_area_states).shape,
        #       np.array(test_area_states[0]).shape)
        # print("***test_data_action", area, ":", np.array(test_area_actions).shape,
        #       np.array(test_area_actions[0]).shape)
        # print("***test_data_target", area, ":", np.array(test_data_reweard).shape)

    #label

    # print('train_target:', np.array(train_target).shape)
    # print('test_target:', np.array(test_target).shape)
    print("***test_data_global_state :", len(train_data_global_state))
    #       np.array(test_area_states[0]).shape)
    return train_data_state,train_data_action,train_data_global_state,train_target,test_data_state,test_data_action,test_data_global_state,test_target

def supervised_train_critics(start_idx=start_idx_set, model_list=None,exp_name = "pre_training_hd6_ELU_HC-" + str(start_idx_set),phase='training'):
    hc_hidden_dim=6
    print("**********Reading the dataset************")
    train_data_state,train_data_action,train_data_global_state,train_target,test_data_state,test_data_action,test_data_global_state,test_data_target=dataset(start_idx)
    test_global_state = merge_dicts(test_data_global_state)
    test_target=torch.FloatTensor(test_data_target[settings.area_name[0]]).to(device)
    critic_list={}
    all_para=[]
    for area in settings.area_name:
        if model_list is not None:
            model=model_list[area]["critic"]
        else:
            state_dim= len(getattr(settings, area + "_branch")) * 9 + len(
                getattr(settings, area + "_gen")) + len(settings.ldname) * 2 + 1
            for local_gen in getattr(settings, area + "_gen"):
                gen_id = settings.name_index[local_gen]
                if gen_id in settings.renewable_ids:
                    state_dim = state_dim + 1

            action_dim= len(getattr(settings, area + "_gen"))
            print(area,"state_dim",state_dim)
            print(area,"action_dim",action_dim)
            model=Dis_critic(state_dim,action_dim)
        critic_list[area]=model.to(device).eval()
        model_path_p = os.path.join("log", exp_name, 'pretrain_model_para', area + "_pretrain_critic.pth")
        if os.path.exists(model_path_p):
            model.load_state_dict(torch.load(model_path_p, map_location=device))
        all_para=all_para+list(model.parameters())
    HC=Hybrid_critic(settings.num_area,hc_hidden_dim).to(device).eval()
    model_path_hc = os.path.join("log", exp_name, 'pretrain_model_para', "pretrain_hyb_critic.pth")
    if os.path.exists(model_path_hc):
        HC.load_state_dict(torch.load(model_path_hc, map_location=device))
    all_para = all_para + list(HC.parameters())
    optimizer = torch.optim.Adam(all_para, lr=LR)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)

    if not os.path.exists(os.path.join("log", exp_name, 'pretrain_model_para')):
        os.makedirs(os.path.join("log", exp_name, 'pretrain_model_para'))
        print("make the new dirs...")
    with open(os.path.join("log", exp_name, 'pretrain_model_para', "pre_training_all_critic_loss.csv"), "a+",
              newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "test_loss"])

    for epoch in range(0, epoch_num):
        ind = np.random.randint(0, len(train_data_global_state), size=BATCH_SIZE)
        s, a, g_s, t = {}, {}, [], []
        for area in settings.area_name:
            s[area]=[]
            a[area]=[]
            for i in ind:
                s[area].append(dc(train_data_state[area][i]))
                a[area].append(dc(train_data_action[area][i]))
                if area==settings.area_name[0]:
                    g_s.append(dc(train_data_global_state[i]))
                    t.append(dc(train_target[area][i]))


        global_state = merge_dicts(g_s)
        target=torch.FloatTensor(t).to(device)
        agent_q=[]
        for area in settings.area_name:
            agent_q.append(critic_list[area](torch.FloatTensor(s[area]).to(device),
                                             torch.FloatTensor(a[area]).to(device)))
        qs_tensor=torch.cat(agent_q,dim=1).view(-1,1,settings.num_area)
        # print(qs_tensor.shape)
        total_q_output= HC(qs_tensor,global_state)
        loss= torch.nn.functional.mse_loss(total_q_output, target,reduction='mean')
        train_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        StepLR.step()

        ####test#####
        test_agent_q=[]
        for area in settings.area_name:
            test_agent_q.append(critic_list[area](torch.FloatTensor(test_data_state[area]).to(device),
                                             torch.FloatTensor(test_data_action[area]).to(device)))
        test_qs_tensor=torch.cat(test_agent_q,dim=1).view(-1,1,settings.num_area)
        test_total_q= HC(test_qs_tensor,test_global_state)
        test_loss= torch.nn.functional.mse_loss(test_total_q, test_target,reduction='mean').item()

        print("Training actor, epoch:", epoch, "----", "train_loss:", train_loss, "test_loss:",
              test_loss)



        if epoch %10==0:
            for area in settings.area_name:
                model_path_p= os.path.join("log", exp_name, 'pretrain_model_para', area + "_pretrain_critic.pth")
                torch.save(critic_list[area].state_dict(), model_path_p)
            model_path_hc = os.path.join("log", exp_name, 'pretrain_model_para', "pretrain_hyb_critic.pth")
            torch.save(HC.state_dict(), model_path_hc)






        with open(os.path.join("log", exp_name, 'pretrain_model_para', "pre_training_all_critic_loss.csv"),"a+",newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch,train_loss,test_loss])

        print("current_lr: %s" % (optimizer.state_dict()['param_groups'][0]['lr']))
        if epoch >= 800 or (train_loss <= 0.1 and test_loss<=0.1):
            for area in settings.area_name:
                model_path_p = os.path.join("log", exp_name, 'pretrain_model_para', area + "_pretrain_critic.pth")
                torch.save(critic_list[area].state_dict(), model_path_p)
            model_path_hc = os.path.join("log", exp_name, 'pretrain_model_para', "pretrain_hyb_critic.pth")
            torch.save(HC.state_dict(), model_path_hc)
            print("Complicate the training of critic!")
            break

    return True

if __name__ == '__main__':
    supervised_train_critics()




