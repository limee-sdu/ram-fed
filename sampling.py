import numpy as np
from torchvision import datasets, transforms
import torch
import copy


def get_user_typep(x, setting_array=[7, 1, 1, 1]):
    a = 10 * setting_array[0]
    b = a + 10 * setting_array[1]
    c = b + 10 * setting_array[2]

    if 0 <= x < a:
        return 1
    elif a <= x < b:
        return 2
    elif b <= x < c:
        return 3
    elif c <= x < 100:
        return 4
    else:
        return -1

def get_local_wmask1(ranks):
    local_masks = []
    opp_local_masks=[]
    mask = copy.deepcopy(ranks) * 0 + 1
    local_masks.append(mask.view(98635, 128))
    opp_local_masks.append(mask.view(98635, 128))
    mask0 = copy.deepcopy(ranks) * 0
    mask1 = copy.deepcopy(ranks) * 0 + 1

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 9468960, x < 12625280), mask0, mask1)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 9468960, x < 12625280), mask1, mask0)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 6312640, x < 9468960), mask0, mask1)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 6312640, x < 9468960), mask1, mask0)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 3156320, x < 6312640), mask0, mask1)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 3156320, x < 6312640), mask1, mask0)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 3156320), mask0, mask1)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 3156320), mask1, mask0)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 6312640, x < 12625280), mask0, mask1)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 6312640, x < 12625280), mask1, mask0)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 3156320, x < 6312640), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 9468960, x < 12625280), mask0, mask)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 3156320, x < 6312640), mask1, mask0)
    mask = torch.where(torch.logical_and(x >= 9468960, x < 12625280), mask1, mask)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 3156320, x < 9468960), mask0, mask1)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 3156320, x < 9468960), mask1, mask0)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 3156320), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 9468960, x < 12625280), mask0, mask)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 3156320), mask1, mask0)
    mask = torch.where(torch.logical_and(x >= 9468960, x < 12625280), mask1, mask)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 3156320), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 6312640, x < 9468960), mask0, mask)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 3156320), mask1, mask0)
    mask = torch.where(torch.logical_and(x >= 6312640, x < 9468960), mask1, mask)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 6312640), mask0, mask1)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 6312640), mask1, mask0)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 3156320, x < 12625280), mask0, mask1)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 3156320, x < 12625280), mask1, mask0)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 3156320), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 6312640, x < 12625280), mask0, mask)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 3156320), mask1, mask0)
    mask = torch.where(torch.logical_and(x >= 6312640, x <12625280), mask1, mask)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 6312640), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 9468960, x < 12625280), mask0, mask)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 6312640), mask1, mask0)
    mask = torch.where(torch.logical_and(x >= 9468960, x < 12625280), mask1, mask)
    opp_local_masks.append(mask.view(98635, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 9468960), mask0, mask1)
    local_masks.append(mask.view(98635, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 9468960), mask1, mask0)
    opp_local_masks.append(mask.view(98635, 128))

    return local_masks,opp_local_masks


def get_local_wmask2(ranks):
    local_masks = []
    opp_local_masks=[]
    mask = copy.deepcopy(ranks) * 0 + 1
    local_masks.append(mask.view(128, 128))
    opp_local_masks.append(mask.view(128, 128))
    mask0 = copy.deepcopy(ranks) * 0
    mask1 = copy.deepcopy(ranks) * 0 + 1

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 12288, x < 16384), mask0, mask1)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 12288, x < 16384), mask1, mask0)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 8192, x < 12288), mask0, mask1)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 8192, x < 12288), mask1, mask0)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 4096, x < 8192), mask0, mask1)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >=4096, x < 8192), mask1, mask0)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 4096), mask0, mask1)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 4096), mask1, mask0)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 8192, x < 16384), mask0, mask1)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 8192, x < 16384), mask1, mask0)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 4096, x < 8192), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 12288, x < 16384), mask0, mask)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 4096, x < 8192), mask1, mask0)
    mask = torch.where(torch.logical_and(x >= 12288, x < 16384), mask1, mask)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >=4096, x < 12288), mask0, mask1)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 4096, x < 12288), mask1, mask0)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 4096), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 12288, x < 16384), mask0, mask)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 4096), mask1, mask0)
    mask = torch.where(torch.logical_and(x >= 12288, x < 16384), mask1, mask)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 4096), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 8192, x < 12288), mask0, mask)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 4096), mask1, mask0)
    mask = torch.where(torch.logical_and(x >= 8192, x < 12288), mask1, mask)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 8192), mask0, mask1)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 8192), mask1, mask0)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 4096, x < 16384), mask0, mask1)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 4096, x < 16384), mask1, mask0)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 4096), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 8192, x < 16384), mask0, mask)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 4096), mask1, mask0)
    mask = torch.where(torch.logical_and(x >=8192, x < 16384), mask1, mask)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 8192), mask0, mask1)
    mask = torch.where(torch.logical_and(x >= 12288, x < 16384), mask0, mask)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 8192), mask1, mask0)
    mask = torch.where(torch.logical_and(x >= 12288, x < 16384), mask1, mask)
    opp_local_masks.append(mask.view(128, 128))

    x = copy.deepcopy(ranks)
    mask = torch.where(torch.logical_and(x >= 0, x < 12288), mask0, mask1)
    local_masks.append(mask.view(128, 128))
    mask = torch.where(torch.logical_and(x >= 0, x < 12288), mask1, mask0)
    opp_local_masks.append(mask.view(128, 128))
    return local_masks,opp_local_masks


def get_local_wmasks(ranks1,ranks2):
    local_masks1,opp_local_masks1=get_local_wmask1(ranks1)
    local_masks2,opp_local_masks2=get_local_wmask2(ranks2)
    return local_masks1,opp_local_masks1, local_masks2,opp_local_masks2

def get_mat(p_array, idx):
    x = np.ones(200)
    if idx == 1:
        return x
    for i in range(len(p_array)):
        if p_array[i] == idx:
            x[i] = 0
    return x


def get_matrxs(p_array):
    x = []
    for i in range(1, 5):
        x.append(get_mat(p_array, i))
    return x


def get_onehot_matrixs(rank):
    p_array = get_Pmat(rank)
    x = get_matrxs(p_array)
    return x

def get_Pmat(rank):
    def get_P(x):
        if x >= 0 and x < 50:
            return 1
        elif x > 49 and x < 100:
            return 2
        elif x > 99 and x < 150:
            return 3
        elif x > 149 and x < 200:
            return 4
        else:
            return -1

    x = np.zeros(200)
    for i in range(len(rank)):
        x[i] = get_P(rank[i])
    return x


def mnist_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
