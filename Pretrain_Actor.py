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
import os
import sys
import sys
import os
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(parent_dir)
from Dispatching_Agents.Actor import Dis_actor
from utilize.settings import settings


start_idx_set = 42312
BATCH_SIZE=64
LR = 0.001
epoch_num = 100000000000000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str_list_to_float_list(str_list):
    n = 0
    while n < len(str_list):
        str_list[n] = float(str_list[n])
        n += 1
    return(str_list)


def dataset(start_idx_set,train_data_len=800,test_data_len=600):
    pretrain_start = start_idx_set-100
    train_data_end = pretrain_start+train_data_len#分割数据集


    # pretrain_start = 0
    # train_data_end = 100000  # 分割数据集

    test_start = start_idx_set + 1
    test_data_end = test_start + test_data_len

    # pretrain_start = 0
    # train_data_size = (pretrain_start+106820) * train_data#分割数据集
    train_data = {}
    test_data={}
    train_target = {}
    test_target={}
#actor输入
    train_data_a_ex =[]
    train_data_p_ex =[]
    train_data_v_ex =[]
    train_data_q_ex =[]
    train_data_a_or=[]
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

            if i>test_start and i<=test_data_end:
                row = rows
                row = str_list_to_float_list(row)
                test_data_gen_p.append(row)
            if i>test_data_end and i>train_data_end:
                break

    for area in settings.area_name:
        train_area_states=[]
        train_area_targets = []
        for i in range(0,len(train_data_gen_q)):
            state=[]
            for local_branch in getattr(settings, area + "_branch"):
                branch_id = settings.lnname.index(local_branch)
                state=  state +[train_data_a_ex[i][branch_id]] + [train_data_p_ex[i][branch_id]] + [train_data_q_ex[i][branch_id]] + [train_data_v_ex[i][branch_id]] + \
                        [train_data_a_or[i][branch_id]] +[train_data_p_or[i][branch_id]] + [train_data_q_or[i][branch_id]] + [train_data_v_or[i][branch_id]] + \
                        [train_data_rho[i][branch_id]]
            for local_gen in getattr(settings, area + "_gen"):
                gen_id = settings.name_index[local_gen]
                state = state + [train_data_gen_q[i][gen_id]]
                if gen_id in settings.renewable_ids:
                    state = state +[train_data_max_p[i][settings.renewable_ids.index(gen_id)]]

            state = state + train_data_grid_loss[i] + train_data_load_p[i] + train_data_load_q[i]
            train_area_states.append(state)

            train_area_target = []
            for local_gen in getattr(settings, area + "_gen"):
                gen_id = settings.name_index[local_gen]
                train_area_target.append(train_data_gen_p[i][gen_id])
            train_area_targets.append(train_area_target)
        print("***train_data_state", area,":",np.array(train_area_states).shape,np.array(train_area_states[0]).shape)
        train_data[area]=train_area_states
        print("***train_data_target", area, ":", np.array(train_area_targets).shape,
              np.array(train_area_targets[0]).shape)
        train_target[area] = train_area_targets

        test_area_states = []
        test_area_targets = []
        for i in range(0,len(test_data_gen_q)):#
            state = []
            for local_branch in getattr(settings, area + "_branch"):
                branch_id = settings.lnname.index(local_branch)
                state = state + [test_data_a_ex[i][branch_id]] + [test_data_p_ex[i][branch_id]] + [test_data_q_ex[i][branch_id]] + \
                        [test_data_v_ex[i][branch_id]] + [test_data_a_or[i][branch_id]] + [test_data_p_or[i][branch_id]] + \
                        [test_data_q_or[i][branch_id]] + [test_data_v_or[i][branch_id]] + [test_data_rho[i][branch_id]]
            for local_gen in getattr(settings, area + "_gen"):
                gen_id = settings.name_index[local_gen]
                state = state + [test_data_gen_q[i][gen_id]]
                if gen_id in settings.renewable_ids:
                    state = state + [test_data_max_p[i][settings.renewable_ids.index(gen_id)]]

            state = state + test_data_grid_loss[i] + test_data_load_p[i] + test_data_load_q[i]
            test_area_states.append(state)

            test_area_target = []
            for local_gen in getattr(settings, area + "_gen"):
                gen_id = settings.name_index[local_gen]
                test_area_target.append(test_data_gen_p[i][gen_id])
            test_area_targets.append(test_area_target)
        # print("***test_data_state", area, ":", np.array(test_area_states).shape, np.array(test_area_states[0]).shape)
        test_data[area] = test_area_states
        test_target[area] = test_area_targets

    # for area in settings.area_name:
    #     train_area_targets=[]
    #     for i in range(0,len(train_data_gen_p)):
    #         train_area_target = []
    #         for local_gen in getattr(settings, area + "_gen"):
    #             gen_id = settings.name_index[local_gen]
    #             train_area_target.append(train_data_gen_p[i][gen_id])
    #         train_area_targets.append(train_area_target)
    #     print("***train_data_target", area, ":", np.array(train_area_targets).shape, np.array(train_area_targets[0]).shape)
    #     train_target[area]=train_area_targets
    #
    #     test_area_targets = []
    #     for i in range(0, len(test_data_gen_p)):
    #         test_area_target = []
    #         for local_gen in getattr(settings, area + "_gen"):
    #             gen_id = settings.name_index[local_gen]
    #             test_area_target.append(test_data_gen_p[i][gen_id])
    #         test_area_targets.append(test_area_target)
    #     # print("***test_data_target", area, ":", np.array(test_area_targets).shape, np.array(test_area_targets[0]).shape)
    #     test_target[area] = test_area_targets
    # print('train_target:', np.array(train_target).shape)
    # print('test_target:', np.array(test_target).shape)
    return train_data,train_target,test_data,test_target


def fit(epoch,model,data_loader,optimizer,test_data,test_target,phase='training'):
    if phase == "training":  # 判断当前是训练还是验证
        model.train()
    if phase == "validation":
        model.eval()
    all_loss=0.0
    times=0
    for batch_idx,(data_,target_) in enumerate(data_loader):
        # print('data[i]',data[i])
        data, target = data_.to(device), target_.to(device)  # 使用cuda加速
        #target.argmax(1)
        #data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()  # 重置梯度
        output = model(data)  # 得出预测结果

        loss = torch.nn.functional.mse_loss(output, target,reduction='mean')  # 计算损失值

        all_loss+=torch.nn.functional.mse_loss(output, target,reduction='mean').item()

        if phase == 'training':
            loss.backward()
            optimizer.step()
            times=times+1

    test_output= model(test_data)
    test_loss = torch.nn.functional.mse_loss(test_output, test_target,reduction='mean').item()
    print("Training actor, epoch:",epoch,"----","train_loss:",all_loss/times,"test_loss:",test_loss)

    return all_loss/times,test_loss


def supervised_train_actors(start_idx=start_idx_set, model_list=None,exp_name = "pre_training_actor_ELU_fuzz-" + str(start_idx_set),phase='training'):
    print("**********Reading the dataset************")
    train_data, train_target, test_data, test_target = dataset(start_idx)
    # print(np.array(train_data).shape)
    # print(torch.Tensor(train_data).shape, torch.Tensor(train_target).shape)
    for area in settings.area_name:
        print("training...",area)
        train_dataset = Data.TensorDataset(torch.Tensor(train_data[area]), torch.Tensor(train_target[area]))
        if model_list is not None:
            model=model_list[area]["actor"]
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
            model=Dis_actor(state_dim,action_dim)

        model_path_p = os.path.join("log", exp_name, 'pretrain_model_para', area + "_pretrain_actor.pth")
        if os.path.exists(model_path_p):
            model.load_state_dict(torch.load(model_path_p, map_location=device))

        train_loader = Data.DataLoader(
            dataset=train_dataset,  # torch TensorDataset format
            batch_size=BATCH_SIZE,  # mini batch size
            shuffle=True,  # 要不要打乱数据 (打乱比较好)
            num_workers=2,  # 多线程来读数据
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        if not os.path.exists(os.path.join("log", exp_name, 'pretrain_model_para')):
            os.makedirs(os.path.join("log", exp_name, 'pretrain_model_para'))
            print("make the new dirs...")

        with open(os.path.join("log", exp_name, 'pretrain_model_para', area+"_pre_training_actor_loss.csv"), "a+", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "train_loss","test_loss"])
        # with open(os.path.join('best_model_10', exp_name, "test_result_a.csv_10"),"w",newline="") as csvfile:
        #    writer = csv.writer(csvfile)
        #    writer.writerow(["epoch","loss","accuracy"])

        for epoch in range(0, epoch_num):
            train_loss, test_loss= fit(epoch, model, train_loader, optimizer, test_data=torch.Tensor(test_data[area]),test_target=torch.Tensor(test_target[area]),phase=phase)
            StepLR.step()

            if epoch % 10 == 0:
                torch.save(model.state_dict(), model_path_p)

            with open(os.path.join("log", exp_name, 'pretrain_model_para', area+"_pre_training_actor_loss.csv"), "a+",
                      newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch, train_loss, test_loss])

            # train_losses.append(epoch_loss)
            print("current_lr: %s" % (optimizer.state_dict()['param_groups'][0]['lr']))
            if epoch >= 200 or (test_loss < 20 and train_loss < 20):
                torch.save(model.state_dict(), model_path_p)
                print("Complicate the training of",area, "_actor!")
                break

    return True





        # 把 dataset 放入 DataLoader

        

if __name__ == '__main__':
    # dataset()
   result = supervised_train_actors()






