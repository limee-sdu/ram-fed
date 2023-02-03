import copy
import torch
from torch import nn


def FedAvg(w,type_array,local_w_mask1,local_w_mask2,local_w_mask3):
    w_avg = copy.deepcopy(w[0])
    all_w_mask1,all_w_mask2,all_w_mask3 = get_all_masks(type_array,local_w_mask1,local_w_mask2,local_w_mask3)
    mask1 = copy.deepcopy(all_w_mask1) * 0 + 1
    mask2 = copy.deepcopy(all_w_mask2) * 0 + 1
    mask3 = copy.deepcopy(all_w_mask3) * 0 + 1
    keys = list(w_avg.keys())
    k = keys[0]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k] , len(w))

    k = keys[1]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k] , len(w))

    k = keys[2]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k], all_w_mask1)

    k = keys[3]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k] , len(w))

    k = keys[4]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k], all_w_mask2)
    k = keys[5]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k] , len(w))
    k = keys[6]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k], all_w_mask3)
    for k in keys[7:]:
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    for e in w_avg:
        w_avg[e] = torch.nan_to_num(w_avg[e], nan = 0)
    return w_avg


def get_all_masks(type_array,local_w_masks1,local_w_masks2,local_w_masks3):
    all_w_mask1 = local_w_masks1[0] *0
    all_w_mask2 = local_w_masks2[0] *0

    for e in type_array:
        all_w_mask1 += local_w_masks1[e - 1]
        all_w_mask2 += local_w_masks2[e - 1]
    return all_w_mask1,all_w_mask2

def FedAvg2(w_glob,w,type_array,local_w_mask1,local_w_mask2,local_w_mask3):
    w_avg = copy.deepcopy(w[0])
    all_w_mask1,all_w_mask2,all_w_mask3 = get_all_masks(type_array,local_w_mask1,local_w_mask2,local_w_mask3)
    mask1 = copy.deepcopy(all_w_mask1) * 0 + 1
    mask2 = copy.deepcopy(all_w_mask2) * 0 + 1
    mask3 = copy.deepcopy(all_w_mask3) * 0 + 1
    keys = list(w_avg.keys())
    k = keys[0]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k], len(w))
    k = keys[1]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k], len(w))
    k=keys[2]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.where(w_avg[k]==0, w_glob[k], w_avg[k])
    all_w_mask1= torch.where(all_w_mask1==0, mask1, all_w_mask1)
    w_avg[k] = torch.div(w_avg[k], all_w_mask1)
    k = keys[3]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k], len(w))
    k=keys[4]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.where(w_avg[k]==0, w_glob[k], w_avg[k])
    all_w_mask2= torch.where(all_w_mask2==0, mask2, all_w_mask2)
    w_avg[k] = torch.div(w_avg[k], all_w_mask2)
    k = keys[5]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.div(w_avg[k], len(w))
    k=keys[6]
    for i in range(1, len(w)):
        w_avg[k] += w[i][k]
    w_avg[k] = torch.where(w_avg[k]==0, w_glob[k], w_avg[k])
    all_w_mask3= torch.where(all_w_mask3==0, mask3, all_w_mask3)
    w_avg[k] = torch.div(w_avg[k], all_w_mask3)

    for k in keys[7:]:
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg3(w_glob,w,type_array,local_w_mask1,local_w_mask2):
    w_avg = copy.deepcopy(w[0])
    for k in range(0,5):
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg