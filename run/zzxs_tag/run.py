import tensorflow as tf
import numpy as np
import time
import pickle as pkl
import sys
import os
import shutil

root_path = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, root_path)

from all_in_one.utils import smart_show
from all_in_one.utils import evaluate_func
from all_in_one.utils import export_func

from all_in_one.config import config_utils
from all_in_one.models.cnn_model_one_layer_tag import model_cnn as main_model
from all_in_one.models.cnn_model_one_layer_tag import utilizer

np.random.seed(3306)

np.set_printoptions(linewidth=200)

mission = 'zzxs_tag'
config = config_utils.Config()(root_path + '/all_in_one/config/' + mission + '/config.ini')['model_parameter']


def train(train_data_path, test_data_path, path_prefix, fold):
    embedding_path = path_prefix + '_word_embedding.pkl'
    word_dict_path = path_prefix + '_vocab_inword.pkl'
    label_dict_path = path_prefix + '_label_class_mapping.pkl'

    batch_size = int(config['batch_size'])
    checkpoint_num = int(config['checkpoint_num'])
    timedelay_num = int(config['timedelay_num'])

    print('Loading Data...')

    dictionary_data = pkl.load(open(word_dict_path, 'rb'))
    dictionary = {v: k for k, v in dictionary_data.items()}

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
    maxlen = int((embeddings['maxlen'] - 1) / 2)

    W_train, T_train, Y_train, L_train = utilizer.get_train_data(train_data, dictionary)
    W_dev, T_dev, Y_dev, L_dev = utilizer.get_train_data(dev_data, dictionary)
    W_test, T_test, Y_test, L_test = utilizer.get_train_data(test_data, dictionary)

    is_multilabel = False
    for y in Y_train:
        if sum(y) > 1:
            is_multilabel = True
            break

    if not is_multilabel:
        Y_train = [[1 - sum(y)] + y for y in Y_train]
        Y_dev = [[1 - sum(y)] + y for y in Y_dev]
        Y_test = [[1 - sum(y)] + y for y in Y_test]

    label_class = len(Y_train[0])
    tag_class = len(T_train[0][0]) * 2

    W_train = np.asarray(W_train)
    T_train = np.asarray(T_train)
    L_train = np.asarray(L_train)
    Y_train = np.asarray(Y_train)

    W_dev = np.asarray(W_dev)
    T_dev = np.asarray(T_dev)
    L_dev = np.asarray(L_dev)
    Y_dev = np.asarray(Y_dev)

    W_test = np.asarray(W_test)
    T_test = np.asarray(T_test)
    L_test = np.asarray(L_test)
    Y_test = np.asarray(Y_test)

    print('W_train shape:', W_train.shape)
    print('T_train shape:', T_train.shape)
    print('L_train shape:', L_train.shape)
    print('Y_train shape:', Y_train.shape)

    print('W_dev shape:', W_dev.shape)
    print('T_dev shape:', T_dev.shape)
    print('L_dev shape:', L_dev.shape)
    print('Y_dev shape:', Y_dev.shape)

    print('W_test shape:', W_test.shape)
    print('T_test shape:', T_test.shape)
    print('L_test shape:', L_test.shape)
    print('Y_test shape:', Y_test.shape)

    print('W_embedding shape:', W_embedding.shape)
    print('maxlen:', maxlen)
    print('Label class:', label_class)
    print('Tag class:', tag_class)

    Y_distribution = {}
    for y in Y_train:
        for i, ii in enumerate(y):
            if i not in Y_distribution:
                Y_distribution[i] = 0
            if ii == 1:
                Y_distribution[i] += 1
    Y_distribution_r = {i: np.log(1.0 * (len(Y_train) - v) / v + 1) for i, v in Y_distribution.items()}
    # Y_distribution_r = {i: np.sqrt(1.0 * (len(Y_train) - v) / v + 1) for i, v in Y_distribution.items()}

    Y_distribution_config = [Y_distribution_r[i] for i in range(len(Y_distribution_r))]
    print('Y_distribution_r', Y_distribution_config)

    tf.reset_default_graph()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    model = main_model.Model(label_class, maxlen, W_embedding, label_dict, Y_distribution_config, tag_class, config,
                             multilabel=is_multilabel)

    sess.run(tf.global_variables_initializer())

    best_label = 0
    best_loss = 100
    best_recall = 0
    best_precision = 0
    test_label = 0
    test_loss = 0
    test_recall = 0
    test_precision = 0
    batches = 0
    timedelay = 0
    start_time = time.time()
    i = 0
    while timedelay < timedelay_num:
        if i > len(W_train):
            i = 0

        w_batch = W_train[i: i + batch_size]
        t_batch = T_train[i: i + batch_size]
        sl_batch = L_train[i: i + batch_size]
        y_batch = Y_train[i: i + batch_size]

        # model.train(w_batch, t_batch, sl_batch, y_batch)
        # model.temp_check(sess, w_batch, t_batch, sl_batch, y_batch)
        # model.train(w_batch, t_batch, sl_batch, y_batch, y_distribution_batch)
        # test_loss, test_results = model.test(sess, W_test, T_test, L_test, Y_test, int(batch_size / 1))
        # test_results = smart_show.smart_show(test_results, multilabel=is_multilabel)
        # test_label, test_precision, test_recall = evaluate_func.evaluate_func(test_results, multilabel=is_multilabel)
        # export_func.export(model, sess)
        # print(test_loss)
        # exit()

        i += batch_size
        batches += 1

        model.train(w_batch, t_batch, sl_batch, y_batch)

        if batches % checkpoint_num == 0:
            train_loss, train_results = model.test(sess, w_batch, t_batch, sl_batch, y_batch, batch_size)
            train_label, train_precision, train_recall = evaluate_func.evaluate_func(smart_show.smart_show(train_results,
                                                                                                           multilabel=is_multilabel),
                                                                                     multilabel=is_multilabel)
            train_label = train_label[0]
            train_recall = train_recall[0]
            train_precision = train_precision[0]

            dev_loss, dev_results = model.test(sess, W_dev, T_dev, L_dev, Y_dev, int(batch_size / 1))
            dev_label, dev_precision, dev_recall = evaluate_func.evaluate_func(smart_show.smart_show(dev_results,
                                                                                                     multilabel=is_multilabel),
                                                                                     multilabel=is_multilabel)
            dev_label = dev_label[0]
            dev_recall = dev_recall[0]
            dev_precision = dev_precision[0]

            if best_loss > dev_loss or best_label < dev_label:
                timedelay = 0
                test_loss, test_results = model.test(sess, W_test, T_test, L_test, Y_test, int(batch_size / 1))
                test_label, test_precision, test_recall = evaluate_func.evaluate_func(smart_show.smart_show(test_results,
                                                                                                            multilabel=is_multilabel),
                                                                                   multilabel=is_multilabel)

                test_label = test_label[0]
                test_recall = test_recall[0]
                test_precision = test_precision[0]

                if os.path.exists(root_path + '/all_in_one/demo/exported_models/' + mission + '/' + str(fold + 1)):
                    shutil.rmtree(root_path + '/all_in_one/demo/exported_models/' + mission + '/' + str(fold + 1))
                export_func.export(model, sess, signature_name=mission, version=fold + 1)

            else:
                timedelay += 1

            if best_label < dev_label:
                best_label = dev_label
            if best_loss > dev_loss:
                best_loss = dev_loss
            if best_recall < dev_recall:
                best_recall = dev_recall
            if best_precision < dev_precision:
                best_precision = dev_precision

            sys.stdout.write('Batches: %d' % batches)
            sys.stdout.write('\tBatch Time: %.4fs' % (1.0 * (time.time() - start_time) / checkpoint_num))

            sys.stdout.write('\nTrain Label: %.6f' % train_label)
            sys.stdout.write('\tTrain Loss: %.6f' % train_loss)
            sys.stdout.write('\tTrain Recall: %.6f' % train_recall)
            sys.stdout.write('\tTrain Precision: %.6f' % train_precision)

            sys.stdout.write('\nValid Label: %.6f' % dev_label)
            sys.stdout.write('\tValid Loss: %.6f' % dev_loss)
            sys.stdout.write('\tValid Recall: %.6f' % dev_recall)
            sys.stdout.write('\tValid Precision: %.6f' % dev_precision)

            sys.stdout.write('\nTest Label: %.6f' % test_label)
            sys.stdout.write('\tTest Loss: %.6f' % test_loss)
            sys.stdout.write('\tTest Recall: %.6f' % test_recall)
            sys.stdout.write('\tTest Precision: %.6f' % test_precision)

            sys.stdout.write('\nBest Label: %.6f' % best_label)
            sys.stdout.write('\tBest Loss: %.6f' % best_loss)
            sys.stdout.write('\tBest Recall: %.6f' % best_recall)
            sys.stdout.write('\tBest Precision: %.6f' % best_precision)

            sys.stdout.write('\n\n')

            start_time = time.time()

    return test_label, test_recall, test_precision


def main(_):
    accs_label = []
    recalls_label = []
    precisions_label = []

    path_prefix = root_path + '/all_in_one/data/data_train/' + mission + '/' + mission

    for i in range(10):
        train_data_path = path_prefix + '_cv/train_data_cv_' + str(i) + '.txt'
        test_data_path = path_prefix + '_cv/test_data_cv_' + str(i) + '.txt'
        acc_label, recall_label, precision_label = train(train_data_path, test_data_path, path_prefix, i)
        accs_label.append(acc_label)
        recalls_label.append(recall_label)
        precisions_label.append(precision_label)

        print('Accuracy:')
        print(np.mean(accs_label))
        print(accs_label)
        print('Recall:')
        print(np.mean(recalls_label))
        print(recalls_label)
        print('Precision:')
        print(np.mean(precisions_label))
        print(precisions_label)

    f_log = open(root_path + '/all_in_one/logs/' + mission + '_results.log', 'a')
    f_log.write('Accuracy:\n')
    f_log.write(str(np.mean(accs_label)))
    f_log.write('\n')
    f_log.write(str(accs_label))
    f_log.write('\n')
    f_log.write('Recall:\n')
    f_log.write(str(np.mean(recalls_label)))
    f_log.write('\n')
    f_log.write(str(recalls_label))
    f_log.write('\n')
    f_log.write('Precision:\n')
    f_log.write(str(np.mean(precisions_label)))
    f_log.write('\n')
    f_log.write(str(precisions_label))
    f_log.write('\n\n')
    f_log.close()


if __name__ == '__main__':
    tf.app.run()
