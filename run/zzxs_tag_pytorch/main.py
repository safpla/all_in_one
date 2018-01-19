#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import model
import train
import time
import config
import pickle as pkl
from docopt import docopt
import numpy as np
import utilizer
import os
import pandas as pd


parser = argparse.ArgumentParser(description='CNN text classificer')

# learning
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
# data
# parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-kernel-sizes', type=str, default='1,2,3,4,5', help='comma-separated kernel size to use for convolution')

parser.add_argument('-no-static', action='store_true', default=True, help='fix the embedding')
# device
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
# pretrained embedding
parser.add_argument('-pretrained-embedding', type=str, default='Thuc', help='use pretrained word embedding')

parser.add_argument('-nobabeimode', action='store_true', default=True, help='donot use babei[default: True]')

parser.add_argument('-tagbabei', action='store_true', default=False, help='use tagbabei[default: False]')

parser.add_argument('-taglabel', action='store_true', default=False, help='use taglabel[default: False]')

parser.add_argument('-tagsenindex', action='store_true', default=False, help='use tagsenindex[default: False]')

parser.add_argument('-tagsenindexsep', action='store_true', default=False, help='use tagsenindexsep[default: False]')

parser.add_argument('-pi', type=float, default=95, help='newpi = 1. - max([pi/100.0**cur_iter, lb])')

parser.add_argument('-lossdistribution', action='store_true', default=False, help='use lossdistribution[default: False]')

args = parser.parse_args()



def gen_data_iter(train_data_path, test_data_path, path_prefix):
    embedding_path = '../../data/data_train/' + path_prefix + '/' + path_prefix + '_word_embedding.pkl'
    word_dict_path = '../../data/data_train/' + path_prefix + '/' + path_prefix + '_vocab_inword.pkl'
    label_dict_path = '../../data/data_train/' + path_prefix + '/'+ path_prefix + '_label_class_mapping.pkl'

    print('Loading Data...')

    # print(os.ge)
    print('word_dict_path',word_dict_path)

    dictionary_data = pkl.load(open(word_dict_path, 'rb'))  # word -> word_id
    dictionary = {v: k for k, v in dictionary_data.items()} # word_id ->word

    label_dict_data = pkl.load(open(label_dict_path, 'rb'))
    label_dict = {v: k for k, v in label_dict_data.items()}

    train_data = open(train_data_path, 'r').readlines()
    dev_data = train_data[int(len(train_data) * 0.9):]
    train_data = train_data[:int(len(train_data) * 0.9)]
    test_data = open(test_data_path, 'r').readlines()

    embedding_file = open(embedding_path, 'rb')
    embeddings = pkl.load(embedding_file)
    embedding_file.close()

    W_embedding = np.array(embeddings['pretrain']['word_embedding'], dtype=np.float32)
    maxlen = embeddings['maxlen']

    # maxlen = 50

    W_train, Y_train, L_train, W_train_tag = utilizer.get_train_data(train_data, maxlen)
    W_dev, Y_dev, L_dev, W_dev_tag= utilizer.get_train_data(dev_data, maxlen)
    W_test, Y_test, L_test, W_test_tag= utilizer.get_train_data(test_data, maxlen)

    is_multilabel = False
    for y in Y_train:
        if sum(y) > 1:
            is_multilabel = True
            break

    if not is_multilabel:
        Y_train = [[1 - sum(y)] + y for y in Y_train]
        Y_dev = [[1 - sum(y)] + y for y in Y_dev]
        Y_test = [[1 - sum(y)] + y for y in Y_test]
        label_dict = {i+1 : j for i, j in label_dict.items()}
        label_dict[0] = 0


    label_class = len(Y_train[0])

    W_train = np.asarray(W_train)
    L_train = np.asarray(L_train)
    Y_train = np.asarray(Y_train)
    W_train_tag = np.asarray(W_train_tag)

    W_dev = np.asarray(W_dev)
    L_dev = np.asarray(L_dev)
    Y_dev = np.asarray(Y_dev)
    W_dev_tag = np.asarray(W_dev_tag)


    W_test = np.asarray(W_test)
    L_test = np.asarray(L_test)
    Y_test = np.asarray(Y_test)
    W_test_tag = np.asarray(W_test_tag)


    print('W_train shape:', W_train.shape)
    print('W_train_tag shape:', W_train_tag.shape)

    print('L_train shape:', L_train.shape)
    print('Y_train shape:', Y_train.shape)


    print('W_dev shape:', W_dev.shape)
    print('L_dev shape:', L_dev.shape)
    print('Y_dev shape:', Y_dev.shape)

    print('W_test shape:', W_test.shape)
    print('L_test shape:', L_test.shape)
    print('Y_test shape:', Y_test.shape)

    print('W_embedding shape:', W_embedding.shape)
    print('maxlen:', maxlen)
    print('Label class:', label_class)

    Y_distribution = {}
    for y in Y_train:
        for i, ii in enumerate(y):
            if ii == 1:
                if i in Y_distribution:
                    Y_distribution[i] += 1
                else:
                    Y_distribution[i] = 1
    # Y_distribution_r = {i: np.log(1.0 * (len(Y_train) - v) / v + 1) for i, v in Y_distribution.items()}
    Y_distribution_r = {i: np.sqrt(1.0 * (len(Y_train) - v) / v + 1) for i, v in Y_distribution.items()}

    Y_distribution_config = [Y_distribution_r[i] for i in range(len(Y_distribution_r))]
    print('Y_distribution_r', Y_distribution_config)
    print('W_embedding shape:', W_embedding.shape)
    print('maxlen:', maxlen)
    print('Label class:', label_class)
    print('Label dict:', label_dict)

    return ((W_train, Y_train, W_train_tag), (W_dev, Y_dev, W_dev_tag), (W_test, Y_test, W_test_tag), W_embedding, maxlen, label_class, label_dict, dictionary, Y_distribution_config)



def run():
    path_prefix = 'zzxs_babei_tag'
    accs_test = []
    test_cases = []
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    for i in range(10):
        print("\nLoading data...")
        print('\nThis is CV' + str(i))
        # train_data_path = '../../data/' + path_prefix + '_cv/train_data_cv_' + str(i) + '.txt'
        # test_data_path = '../../data/' + path_prefix + '_cv/test_data_cv_' + str(i) + '.txt'

        train_data_path = '../../data/data_train/' + path_prefix + '/' + path_prefix + '_cv/train_data_cv_' + str(i) + '.txt'

        test_data_path = '../../data/data_train/' + path_prefix + '/' + path_prefix + '_cv/test_data_cv_' + str(i) + '.txt'

        train_iter, dev_iter, test_iter, W_embedding, maxlen, label_class_num, label_dict, dictionary, Y_distribution_config= gen_data_iter(train_data_path, test_data_path, path_prefix)
        args.embed_num = W_embedding.shape[0] # vocab size
        args.class_num = label_class_num

        args.cuda = (not args.no_cuda) and torch.cuda.is_available()
        args.usebabeimode = not args.nobabeimode
        args.Y_distribution_config = Y_distribution_config
        # args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


        # model
        if args.snapshot is None:
            if args.pretrained_embedding is not None:
                args.embed_dim = W_embedding.shape[1]
                cnn = model.CNN_Text(args)
                cnn.embed.weight.data = torch.from_numpy(W_embedding)
            else:
                cnn = model.CNN_Text(args)
        else :
            print('\nLoading model from [%s]...' % args.snapshot)
            try:
                cnn = torch.load(args.snapshot)
            except :
                print("Sorry, This snapshot doesn't exist."); exit()

        print("\nParameters:")
        args.pi = args.pi/100.0
        for attr, value in sorted(args.__dict__.items()):
            print("\t{}={}".format(attr.upper(), value))
        if args.cuda:
            cnn = cnn.cuda()

        acc_test, test_case = train.train(train_iter, dev_iter, test_iter, cnn, label_class_num, args)

        accs_test.append(acc_test)
        test_cases.append(test_case)
        print('average test', np.mean(accs_test))
        print('test list', accs_test)
    
    test_x_ = []
    test_x = []
    test_x_str = []
    test_pred = []
    test_target = []
    for test in test_cases:
        test_x_.extend(test[0])
        test_pred.extend(test[1])
        test_target.extend(test[2])
    for sen in test_x_:   
        tmp_x = [dictionary[i] for i in sen if i > 0]
        test_x_str.append(''.join(tmp_x))   
        test_x.append(tmp_x)


    test_pred = [label_dict[i] for i in test_pred]
    test_target = [label_dict[i] for i in test_target]
    df_test = pd.DataFrame(np.array([test_x, test_x_str, test_pred, test_target]).T, columns=['case', 'case_cut', 'pred', 'target'])
    # excelpath = root_path + '/all_in_one/logs/' + mission + '.xlsx'
    excelpath = path_prefix + '_lossdistribution_' + str(args.lossdistribution)+ '_tagbabei_' + str(args.tagbabei) + '_tagsenindex_' + str(args.tagsenindex) + '_taglabel_' + str(args.taglabel) + '_babeimode_' + str(args.usebabeimode) + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')[-8:] + '.xlsx'
    writer = pd.ExcelWriter(excelpath)
    df_test.to_excel(writer,'Sheet1')
    writer.save()
    run_path = excelpath[:-4]
    with open(run_path, 'w') as f:
        f.write('average test\n' + str(np.mean(accs_test)))
        f.write('\n')
        f.write('test list\n' + str(accs_test))


run()
# sst_cnn()
