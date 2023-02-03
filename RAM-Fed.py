import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import data_utils
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, get_user_typep, get_local_wmasks
from utils.options import args_parser
from models.Update import ag_LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, fastText
from models.Fed import FedAvg, FedAvg2, FedAvg3
from models.test import ag_test_img
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from fvcore.nn import parameter_count
import torch
import torch.nn as nn
from torchstat import stat
import time

args = args_parser()
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False


def get_sub_paras(w_glob, wmask1, wmask2):
    w_l = copy.deepcopy(w_glob)
    w_l['embedding.weight'] = w_l['embedding.weight'] * wmask1
    w_l['fc1.weight'] = w_l['fc1.weight'] * wmask2
    return w_l


if __name__ == '__main__':
    start = time.time()
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.epochs = 300
    alpha = args.alpha

    net_glob = fastText(hidden_dim=128, vocab_size=98635, num_classes=args.num_classes).to(args.device)
    print(net_glob)
    inp = torch.torch.randint(0, 98635, (1, 200)).to(args.device)

    net_glob.train()
    net_2 = copy.deepcopy(net_glob)
    net_2.train()
    net_3 = copy.deepcopy(net_glob)
    net_3.train()
    net_4 = copy.deepcopy(net_glob)
    net_4.train()
    net_5 = copy.deepcopy(net_glob)
    net_5.train()
    net_6 = copy.deepcopy(net_glob)
    net_6.train()
    net_7 = copy.deepcopy(net_glob)
    net_7.train()
    net_8 = copy.deepcopy(net_glob)
    net_8.train()
    net_9 = copy.deepcopy(net_glob)
    net_9.train()
    net_10 = copy.deepcopy(net_glob)
    net_10.train()
    net_11 = copy.deepcopy(net_glob)
    net_11.train()
    net_12 = copy.deepcopy(net_glob)
    net_12.train()
    net_13 = copy.deepcopy(net_glob)
    net_13.train()
    net_14 = copy.deepcopy(net_glob)
    net_14.train()
    net_15 = copy.deepcopy(net_glob)
    net_15.train()

    w_glob = net_glob.state_dict()
    print("-------------------------------")
    print(w_glob.keys())
    starting_weights = copy.deepcopy(w_glob)

    wranks1 = torch.argsort(torch.absolute(w_glob['embedding.weight'].view(-1)))
    wranks2 = torch.argsort(w_glob['fc1.weight'].view(-1))

    local_w_masks1, opp_local_w_masks1, local_w_masks2, opp_local_w_masks2 = get_local_wmasks(wranks1, wranks2)

    w_n2 = get_sub_paras(w_glob, local_w_masks1[1], local_w_masks2[1])
    net_2.load_state_dict(w_n2)
    w_n3 = get_sub_paras(w_glob, local_w_masks1[2], local_w_masks2[2])
    net_3.load_state_dict(w_n3)
    w_n4 = get_sub_paras(w_glob, local_w_masks1[3], local_w_masks2[3])
    net_4.load_state_dict(w_n4)
    w_n5 = get_sub_paras(w_glob, local_w_masks1[4], local_w_masks2[4])
    net_5.load_state_dict(w_n5)
    w_n6 = get_sub_paras(w_glob, local_w_masks1[5], local_w_masks2[5])
    net_6.load_state_dict(w_n6)
    w_n7 = get_sub_paras(w_glob, local_w_masks1[6], local_w_masks2[6])
    net_7.load_state_dict(w_n7)
    w_n8 = get_sub_paras(w_glob, local_w_masks1[7], local_w_masks2[7])
    net_8.load_state_dict(w_n8)
    w_n9 = get_sub_paras(w_glob, local_w_masks1[8], local_w_masks2[8])
    net_9.load_state_dict(w_n9)
    w_n10 = get_sub_paras(w_glob, local_w_masks1[9], local_w_masks2[9])
    net_10.load_state_dict(w_n10)
    w_n11 = get_sub_paras(w_glob, local_w_masks1[10], local_w_masks2[10])
    net_11.load_state_dict(w_n11)
    w_n12 = get_sub_paras(w_glob, local_w_masks1[11], local_w_masks2[11])
    net_12.load_state_dict(w_n12)
    w_n13 = get_sub_paras(w_glob, local_w_masks1[12], local_w_masks2[12])
    net_13.load_state_dict(w_n13)
    w_n14 = get_sub_paras(w_glob, local_w_masks1[13], local_w_masks2[13])
    net_14.load_state_dict(w_n14)
    w_n15 = get_sub_paras(w_glob, local_w_masks1[14], local_w_masks2[14])
    net_15.load_state_dict(w_n15)


    setting_50 = [6, 7, 8, 9, 10, 11]
    setting_25 = [12, 13, 14, 15]
    setting_arrays = [
        [15, 15, 12, 12, 13, 13, 12, 12, 14, 14],
    ]

    train_data = []
    datadir = 'agnews_' + str(alpha)
    for i in range(0, args.num_users):
        client_data = data_utils.read_client_data(datadir, i, is_train=True)
        train_data.append(client_data)
    dataset_test = data_utils.read_client_data('agnews', 0, is_train=False)

    avg_test_acc = []
    avg_train_loss = []
    avg_test_loss = []
    for i in range(0, 1):
        net_glob.load_state_dict(starting_weights)
        w_glob = copy.deepcopy(starting_weights)
        loss_train = []
        cv_loss, cv_acc = [], []
        val_loss_pre, counter = 0, 0
        net_best = None
        best_loss = None
        val_acc_list, net_list = [], []

        setting = "setting"
        pic_name = './save/fed_{}_{}_{}_lep{}_iid{}_{}.pdf'.format(args.dataset, args.model, args.epochs, args.local_ep,
                                                                   args.iid, setting)
        txt_name = './save/fed_{}_{}_{}_lep{}_iid{}_{}.txt'.format(args.dataset, args.model, args.epochs, args.local_ep,
                                                                   args.iid, setting)
        npy_name = './save/fed_{}_{}_{}_lep{}_iid{}_{}.npy'.format(args.dataset, args.model, args.epochs, args.local_ep,
                                                                   args.iid, setting)
        net_glob.eval()
        acc_test, loss_test = ag_test_img(net_glob, dataset_test, args)
        iter = -1
        net_glob.train()

        mlp_keys = ['embedding.weight', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
        latest_client_update = []
        start_client_update = []
        for i in range(0, 5):
            start_client_update.append(
                torch.zeros(starting_weights[mlp_keys[i]].size(), dtype=starting_weights[mlp_keys[i]][0].dtype))
        for i in range(args.num_users):
            latest_client_update.append(start_client_update)
        epoch_train_loss = []
        epoch_test_loss = []
        epoch_train_acc = []
        epoch_test_acc = []
        for iter in range(args.epochs):
            setting_array = setting_arrays[iter % 2]

            if iter > 0:
                w_glob = net_glob.state_dict()

                wranks1 = torch.argsort(w_glob['embedding.weight'].view(-1))
                wranks2 = torch.argsort(w_glob['fc1.weight'].view(-1))

                local_w_masks1, opp_local_w_masks1, local_w_masks2, opp_local_w_masks2 = get_local_wmasks(wranks1,
                                                                                                          wranks2)

                w_n2 = get_sub_paras(w_glob, local_w_masks1[1], local_w_masks2[1])
                net_2.load_state_dict(w_n2)
                w_n3 = get_sub_paras(w_glob, local_w_masks1[2], local_w_masks2[2])
                net_3.load_state_dict(w_n3)
                w_n4 = get_sub_paras(w_glob, local_w_masks1[3], local_w_masks2[3])
                net_4.load_state_dict(w_n4)
                w_n5 = get_sub_paras(w_glob, local_w_masks1[4], local_w_masks2[4])
                net_5.load_state_dict(w_n5)
                w_n6 = get_sub_paras(w_glob, local_w_masks1[5], local_w_masks2[5])
                net_6.load_state_dict(w_n6)
                w_n7 = get_sub_paras(w_glob, local_w_masks1[6], local_w_masks2[6])
                net_7.load_state_dict(w_n7)
                w_n8 = get_sub_paras(w_glob, local_w_masks1[7], local_w_masks2[7])
                net_8.load_state_dict(w_n8)
                w_n9 = get_sub_paras(w_glob, local_w_masks1[8], local_w_masks2[8])
                net_9.load_state_dict(w_n9)
                w_n10 = get_sub_paras(w_glob, local_w_masks1[9], local_w_masks2[9])
                net_10.load_state_dict(w_n10)
                w_n11 = get_sub_paras(w_glob, local_w_masks1[10], local_w_masks2[10])
                net_11.load_state_dict(w_n11)
                w_n12 = get_sub_paras(w_glob, local_w_masks1[11], local_w_masks2[11])
                net_12.load_state_dict(w_n12)
                w_n13 = get_sub_paras(w_glob, local_w_masks1[12], local_w_masks2[12])
                net_13.load_state_dict(w_n13)
                w_n14 = get_sub_paras(w_glob, local_w_masks1[13], local_w_masks2[13])
                net_14.load_state_dict(w_n14)
                w_n15 = get_sub_paras(w_glob, local_w_masks1[14], local_w_masks2[14])
                net_15.load_state_dict(w_n15)

            loss_locals = []
            if not args.all_clients:
                w_locals = []

            idxs_users = [i for i in range(0, args.num_users)]
            print(idxs_users)

            type_array = []
            count50 = 0
            count25 = 0
            for id, idx in enumerate(idxs_users):
                if iter == 0:
                    typep = 1
                elif iter % 10 == 0:
                    typep=11
                else:
                    typep=np.random.choice(setting_50)

                local = ag_LocalUpdate(args=args, dataset=train_data[idx])
                new_update = []
                if typep == 1:
                    type_array.append(1)
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_glob[mlp_keys[i]])
                elif typep == 2:
                    type_array.append(2)
                    w, loss = local.train(net=copy.deepcopy(net_2).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[1], local_w_masks2[1])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n2[mlp_keys[i]])
                elif typep == 3:
                    type_array.append(3)
                    w, loss = local.train(net=copy.deepcopy(net_3).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[2], local_w_masks2[2])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n3[mlp_keys[i]])
                elif typep == 4:
                    type_array.append(4)
                    w, loss = local.train(net=copy.deepcopy(net_4).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[3], local_w_masks2[3])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n4[mlp_keys[i]])
                elif typep == 5:
                    type_array.append(5)
                    w, loss = local.train(net=copy.deepcopy(net_5).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[4], local_w_masks2[4])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n5[mlp_keys[i]])
                elif typep == 6:
                    type_array.append(6)
                    w, loss = local.train(net=copy.deepcopy(net_6).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[5], local_w_masks2[5])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n6[mlp_keys[i]])
                elif typep == 7:
                    type_array.append(7)
                    w, loss = local.train(net=copy.deepcopy(net_7).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[6], local_w_masks2[6])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n7[mlp_keys[i]])
                elif typep == 8:
                    type_array.append(8)
                    w, loss = local.train(net=copy.deepcopy(net_8).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[7], local_w_masks2[7])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n8[mlp_keys[i]])
                elif typep == 9:
                    type_array.append(9)
                    w, loss = local.train(net=copy.deepcopy(net_9).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[8], local_w_masks2[8])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n9[mlp_keys[i]])
                elif typep == 10:
                    type_array.append(10)
                    w, loss = local.train(net=copy.deepcopy(net_10).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[9], local_w_masks2[9])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n10[mlp_keys[i]])
                elif typep == 11:
                    type_array.append(11)
                    w, loss = local.train(net=copy.deepcopy(net_11).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[10], local_w_masks2[10])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n11[mlp_keys[i]])
                elif typep == 12:
                    type_array.append(12)
                    w, loss = local.train(net=copy.deepcopy(net_12).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[11], local_w_masks2[11])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n12[mlp_keys[i]])
                elif typep == 13:
                    type_array.append(13)
                    w, loss = local.train(net=copy.deepcopy(net_13).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[12], local_w_masks2[12])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n13[mlp_keys[i]])
                elif typep == 14:
                    type_array.append(14)
                    w, loss = local.train(net=copy.deepcopy(net_14).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[13], local_w_masks2[13])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n14[mlp_keys[i]])
                elif typep == 15:
                    type_array.append(15)
                    w, loss = local.train(net=copy.deepcopy(net_15).to(args.device))
                    w = get_sub_paras(w, local_w_masks1[14], local_w_masks2[14])
                    for i in range(5):
                        new_update.append(w[mlp_keys[i]] - w_n15[mlp_keys[i]])

                if typep > 1:
                    latest_client_update[idx][0] = latest_client_update[idx][0].to(args.device) * opp_local_w_masks1[
                        typep - 1] + new_update[0]
                    latest_client_update[idx][1] = latest_client_update[idx][1].to(args.device) * opp_local_w_masks2[
                        typep - 1] + new_update[1]
                else:
                    latest_client_update[idx][0] = new_update[0]
                    latest_client_update[idx][1] = new_update[1]
                for i in range(2, 5):
                    latest_client_update[idx][i] = new_update[i]
                client_w = []
                for i in range(0, 5):
                    client_w.append(copy.deepcopy(w_glob[mlp_keys[i]] + latest_client_update[idx][i]))

                w_locals.append(copy.deepcopy(client_w))
                w_local = copy.deepcopy(w_glob)
                for i in range(0, 5):
                    w_local[mlp_keys[i]] = client_w[i]
                net_client = copy.deepcopy(net_glob)
                net_client.load_state_dict(w_local)
                local_c = ag_LocalUpdate(args=args, dataset=train_data[idx])
                w_c, loss_c = local_c.train_c(net=copy.deepcopy(net_client).to(args.device))
                loss_locals.append(copy.deepcopy(loss_c))
                loss_locals.append(copy.deepcopy(loss))

            w_g = FedAvg3(w_glob, w_locals, type_array, local_w_masks1, local_w_masks2)
            for i in range(0, 5):
                w_glob[mlp_keys[i]] = w_g[i]
            net_glob.load_state_dict(w_glob)

            loss_avg = sum(loss_locals) / len(loss_locals)
            epoch_train_loss.append(loss_avg)
            loss_train.append(loss_avg)

            net_glob.eval()
            acc_test, loss_test = ag_test_img(net_glob, dataset_test, args)
            epoch_test_loss.append(loss_test)
            epoch_test_acc.append(acc_test.item())
            print('round: ', iter, 'acc: ', acc_test.item())
            net_glob.train()
        avg_train_loss.append(epoch_train_loss)
        avg_test_loss.append(epoch_test_loss)
        avg_test_acc.append(epoch_test_acc)

    avg_epoch_train_loss = []
    avg_epoch_test_loss = []
    avg_epoch_test_acc = []
    for i in range(args.epochs):
        sum = 0
        for j in range(0, 1):
            sum = sum + avg_train_loss[j][i]
        sum = sum / 1
        avg_epoch_train_loss.append(sum)
    for i in range(args.epochs):
        sum = 0
        for j in range(0, 1):
            sum = sum + avg_test_loss[j][i]
        sum = sum / 1
        avg_epoch_test_loss.append(sum)
    with open(txt_name, 'w') as f:
        for i in range(args.epochs):
            print(avg_epoch_train_loss[i], avg_epoch_test_loss[i], avg_epoch_test_acc[i], file=f)
