# coding=utf-8
import tensorflow as tf
import numpy as np
import time
import pickle as pkl
import sys
import os
import shutil
import pandas as pd
import datetime

root_path = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, root_path)

from all_in_one.config import config_utils
from all_in_one.models.cnn_model_one_layer import model_cnn as main_model
from all_in_one.models.cnn_model_one_layer import utilizer
from all_in_one.utils import smart_show
from all_in_one.utils import error_case
from all_in_one.run.focus.index2label import index2label as label2string
np.random.seed(3306)

np.set_printoptions(linewidth=200)

mission = 'focus'
config = config_utils.Config()(root_path + '/all_in_one/config/' + mission + '/config.ini')['model_parameter']

def main(_):
    path_prefix = root_path + '/all_in_one/data/data_train/' + mission + '/' + mission
    checkpoint_dir = root_path + '/all_in_one/demo/exported_models/' + mission + '/' + str(1) + '_batch_size_16-norm_lim_3.0-grad_lim_5.0-filter_num_300/'

    decode_data_path = path_prefix + '_cv/test_data_cv_' + str(0) + '.txt'
    embedding_path = path_prefix + '_word_embedding.pkl'
    word_dict_path = path_prefix + '_vocab_inword.pkl'
    label_dict_path = path_prefix + '_label_class_mapping.pkl'

    batch_size = int(config['batch_size'])
    print('Loading Data...')

    word2index = pkl.load(open(word_dict_path, 'rb'))
    index2word = {v: k for k, v in word2index.items()}
    label2index = pkl.load(open(label_dict_path, 'rb'))
    index2label = {v: k for k, v in label2index.items()}
    label_dict = index2label

    for k, v in index2label.items():
        index2label[k] = label2string[v]

    decode_data = open(decode_data_path, 'r').readlines()

    embedding_file = open(embedding_path, 'rb')
    embeddings = pkl.load(embedding_file)
    embedding_file.close()

    W_embedding = np.array(embeddings['pretrain']['word_embedding'], dtype=np.float32)
    maxlen = embeddings['maxlen']

    W_decode, Y_decode, L_decode = utilizer.get_train_data(decode_data)
    W_decode = np.asarray(W_decode)
    L_decode = np.asarray(L_decode)
    Y_decode = np.asarray(Y_decode)
    print('W_decode:', W_decode.shape)
    print('L_decode:', L_decode.shape)
    print('Y_decode:', Y_decode.shape)
    is_multilabel = True

    label_class = len(Y_decode[0])
    print('label_class:', label_class)

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    Y_distribution = [1] * label_class
    model = main_model.Model(label_class, maxlen, W_embedding, label_dict, Y_distribution, config,
                             multilabel=is_multilabel)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print(checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print('ckpt:', ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise('no model found')

    _, Y_predict = model.test(sess, W_decode, L_decode, Y_decode, int(batch_size / 1))
    Y_predict = smart_show.smart_show(Y_predict, multilabel=is_multilabel)
    features = []
    for line in W_decode:
        features.append(''.join([index2word[x] for x in line]))

    error_case.show_error_case(Y_predict, features, multilabel=is_multilabel,
                          if_print=False, index2label=index2label, output_file='error_case.txt')

if __name__ == '__main__':
    tf.app.run()
