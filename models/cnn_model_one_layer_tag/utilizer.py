import numpy as np


def pad_list(lst, p):
    ll = 0
    for i in range(len(lst)):
        if ll < len(lst[i]):
            ll = len(lst[i])
    for i in range(len(lst)):
        lst[i].extend([p]*(ll-len(lst[i])))
    return np.asarray(lst)


def read_data(l, dictionary):
    tmp_w = []
    tmp_t = []

    tmp_l = l.strip().split('\t')

    tmp_y = [int(y) for y in tmp_l[-1].split(',')]

    tagsplit = 0
    for k, v in dictionary.items():
        if v == 'tagsplit':
            tagsplit = k
            break

    tmp_l = tmp_l[0].split(' ' + str(tagsplit) + ' ')
    tmp_l = [ll.split(' ') for ll in tmp_l]

    for n in range(len(tmp_l[0])):
        if tmp_l[0][n] != '0':
            tmp_w.append(int(tmp_l[0][n]))
            ts = int(tmp_l[1][n])
            ts = dictionary[ts]
            tmp_t.append([int(ts[tt]) + tt * 2 for tt in range(len(ts))])

    for i in range(4):
        tmp_w.insert(0, 0)
        tmp_w.append(0)

        tmp_t.insert(0, [2 * j for j in range(len(tmp_t[0]))])
        tmp_t.append([2 * j for j in range(len(tmp_t[0]))])

    tmp_len = len(tmp_w)

    return tmp_w, tmp_t, tmp_y, tmp_len


def padding(l, length):
    while len(l) < length:
        l.append(0)
    return l


def get_train_data(data, dictionary, maxlen=0):
    W = []
    T = []
    Y = []
    L = []
    for l in data:
        tmp_w, tmp_t, tmp_y, tmp_len = read_data(l, dictionary)
        if maxlen > 0:
            while len(tmp_w) < maxlen:
                tmp_w.append(0)
                tmp_t.append([0] * len(tmp_t[0]))
            tmp_w = tmp_w[:maxlen]
            tmp_t = tmp_t[:maxlen]
        W.append(tmp_w)
        T.append(tmp_t)
        Y.append(tmp_y)
        L.append(tmp_len)

    return W, T, Y, L
