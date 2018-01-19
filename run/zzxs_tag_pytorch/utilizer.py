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
    # l is a line      0 0 0 0 12 34 45 5 0 0 0 0 4 0 0 0 0 89 72 87 0 0 0 0 4 5 5 5 5 6 6 6 5 5 5 5 7 0 1 0\t00001010
    # if 4('\t\t\t') in l, means that the number behind 4 is ba/bei word
    # return {'padding': 0, 'OOV': 1, 'SOS': 2, 'EOS': 3, '\t\t\t': 4, 'nobabei': 5, 'babei': 6,
    #         'tagsplit':7, 'tag11':8, 'tag10':9, 'tag21':10, 'tag20':11,'tag31':12, 'tag30':13,
    #         '000':14, '001':15, '010':16, '011':17, '100':18, '101':19,'110':20, '111':21}
    # the number after 7 is tag

    tmp_w = []
    tmp_but_w = []
    tmp_l = l.strip().split('\t')

    case = tmp_l[:-1][0].split(' 4 ')[0]

    # to get the sen index of the tag
    tmp_sen_index_tag = tmp_l[:-1][0].split(' 4 ')[-1]
    tmp_sen_index_tag = [int(i)-5 for i in tmp_sen_index_tag.split(' ')]
    # 5-12  -> 0-7


    cut_t = case.split(' 4 ')

    for c in case.split(' '):
        tmp_w.append(int(c))

    tmp_y = [int(y) for y in tmp_l[-1].split(',')]

    tmp_len = len(tmp_w)
    x_1 = []
    x_2 = []
    x_3 = []
    for i in tmp_sen_index_tag:
        if i in [0,1,2,3]:
            x_1.append(0)
        else:
            x_1.append(1)
        if i in [0,1,4,5]:
            x_2.append(0)
        else:
            x_2.append(1)
        if i in [0,2,4,6]:
            x_3.append(0)
        else:
            x_3.append(1)
    tmp_sen_index_tag = [x_1, x_2, x_3]

    return tmp_w, tmp_y, tmp_len, tmp_sen_index_tag


def padding(l, length):
    while len(l) < length:
        l.append(0)
    return l


def get_train_data(data, maxlen):
    W = []
    Y = []
    L = []
    sen_index_tag = []
    tag_1 = []
    tag_2 = []
    tag_3 = []
    for l in data:
        tmp_w, tmp_y, tmp_len, tmp_tag = read_data(l)
        W.append(tmp_w)
        Y.append(tmp_y)
        L.append(tmp_len)
        tag_1.append(tmp_tag[0])
        tag_2.append(tmp_tag[1])
        tag_3.append(tmp_tag[2])
    sen_index_tag = [tag_1, tag_2, tag_3]
    for y in range(len(Y)):
        Y[y] = [j for j in Y[y]]
    # print('W', type(W))
    # print('but_w', type(but_w))

    return W, Y, L, sen_index_tag