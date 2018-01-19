import numpy as np


def pad_list(lst):
    ll = 0
    for i in range(len(lst)):
        if ll < len(lst[i]):
            ll = len(lst[i])
    for i in range(len(lst)):
        lst[i].extend([0]*(ll-len(lst[i])))
    return np.asarray(lst)


def read_data(l):
    tmp_w = []

    tmp_l = l.strip().split('\t')

    for n in tmp_l[:-1]:
        n = n.split(' ')
        for c in n:
            if c != '0':
                tmp_w.append(int(c))

    tmp_y = [int(y) for y in tmp_l[-1].split(',')]

    for i in range(4):
        tmp_w.insert(0, 0)
        tmp_w.append(0)

    tmp_len = len(tmp_w)

    return tmp_w, tmp_y, tmp_len


def padding(l, length):
    while len(l) < length:
        l.append(0)
    return l


def get_train_data(data, maxlen=0):
    W = []
    Y = []
    L = []
    for l in data:
        tmp_w, tmp_y, tmp_len = read_data(l)
        if maxlen > 0:
            while len(tmp_w) < maxlen:
                tmp_w.append(0)
            tmp_w = tmp_w[:maxlen]
        W.append(tmp_w)
        Y.append(tmp_y)
        L.append(tmp_len)

    return W, Y, L