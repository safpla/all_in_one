import tensorflow as tf
import numpy as np
import time
import pickle as pkl
import sys
import os
import shutil
import pandas as pd
import datetime
import json

root_path = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, root_path)

from all_in_one.utils import smart_show
from all_in_one.utils import evaluate_func
from all_in_one.utils import export_func

from all_in_one.config import config_utils
from all_in_one.models.cnn_model_hierarchical_supervision import model_cnn_hierarchical_supervision as main_model

from focus.utils.compare import main as compare_fn

np.random.seed(3306)

np.set_printoptions(linewidth=200)

mission = 'focus_hierarchical_supervision'
mission_data = 'focus_hierarchical_supervision'

def train(train_data_path, valid_data_path, test_data_path, path_prefix, config,
          load_model=None):
    embedding_path = path_prefix + mission_data + '_word_embedding.pkl'
    word_dict_path = path_prefix + mission_data + '_vocab_inword.pkl'
    label_dict_path = path_prefix + mission_data + '_label_class_mapping.pkl'

    batch_size = int(config['batch_size'])
    checkpoint_num = int(config['checkpoint_num'])
    timedelay_num = int(config['timedelay_num'])
    y_dis_mode = config['y_dis_mode']
    print('Loading Data...')

    dictionary_data = pkl.load(open(word_dict_path, 'rb'))
    dictionary = {v: k for k, v in dictionary_data.items()}

    label_dict_data = pkl.load(open(label_dict_path, 'rb'))
    label_dict = {v: k for k, v in label_dict_data.items()}
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)
        train_data_temp = []
        for data in train_data:
            if data['dfdt_input'] != [] and data['court_input'] != []:
                train_data_temp.append(data)
        train_data = train_data_temp

    with open(valid_data_path, 'r') as f:
        valid_data = json.load(f)
        valid_data_temp = []
        for data in valid_data:
            if data['dfdt_input'] != [] and data['court_input'] != []:
                valid_data_temp.append(data)
        valid_data = valid_data_temp

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        test_data_temp = []
        for data in test_data:
            if data['dfdt_input'] != [] and data['court_input'] != []:
                test_data_temp.append(data)
        test_data = test_data_temp

    embedding_file = open(embedding_path, 'rb')
    embeddings = pkl.load(embedding_file)
    embedding_file.close()

    W_embedding = np.array(embeddings['pretrain']['word_embedding'], dtype=np.float32)
    maxlen = embeddings['maxlen']

    is_multilabel = True
    num_classes = len(train_data[0]['dfdt_label'][0])

    print('train length:', len(train_data))
    print('dev length:', len(valid_data))
    print('test length:', len(test_data))

    print('W_embedding shape:', W_embedding.shape)
    print('Maxlen:', maxlen)
    print('Num_classes from data:', num_classes)

    Y_distribution = {}
    labels = []
    for feature in train_data:
        dfdt_label = feature['dfdt_label']
        court_label = feature['court_label']
        labels.extend(dfdt_label)
        labels.extend(court_label)
    for y in labels:
        for i, ii in enumerate(y):
            if i not in Y_distribution:
                Y_distribution[i] = 1
            if ii == 1:
                Y_distribution[i] += 1
    if y_dis_mode == 'log':
        Y_distribution_r = {i: np.log(1.0 * (len(labels) - v) / v + 1) for i, v in Y_distribution.items()}
        Y_distribution_config = [Y_distribution_r[i] for i in range(len(Y_distribution_r))]
    elif y_dis_mode == 'sqrt':
        Y_distribution_r = {i: np.sqrt(1.0 * (len(Y_train) - v) / v + 1) for i, v in Y_distribution.items()}
        Y_distribution_config = [Y_distribution_r[i] for i in range(len(Y_distribution_r))]
    elif y_dis_mode == 'six':
        Y_distribution_r = {i: np.sqrt(1.0 * (len(Y_train) - v) / v + 1) for i, v in Y_distribution.items()}
        Y_distribution_config = [Y_distribution_r[i] for i in range(len(Y_distribution_r))]
        major_six = [1, 2, 4, 11, 14, 15]
        for i in range(num_classes):
            if i not in major_six:
                Y_distribution_config[i] = 0
    else:
        Y_distribution_config = [1] * num_classes
    print('Y_distribution_r', Y_distribution_config)

    tf.reset_default_graph()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    model = main_model.Model(W_embedding, Y_distribution_config, config,
                             multilabel=is_multilabel)

    sess.run(tf.global_variables_initializer())
    if load_model:
        model.load_model(sess, load_model)
    saver = tf.train.Saver(max_to_keep=10)
    model_name = 'batch_size_{}-filter_num_{}-filter_lengths_{}-dfdt_only_{}-lossweights_{}-sepa_conv_{}-class{}-pp_{}-y_dis_{}-round{}-data18'.format(
        config['batch_size'],
        config['filter_num'],
        config['filter_lengths'],
        config['dfdt_only'],
        config['loss_weights'],
        config['sepa_conv'],
        config['iClass'],
        config['layer_postprocess_sequence'],
        config['y_dis_mode'],
        config['round_num'])

    export_dir_ = root_path + '/all_in_one/demo/exported_models/' + mission
    if not os.path.exists(export_dir_):
        os.mkdir(export_dir_)

    export_dir_ = export_dir_ + '/{}/'.format(model_name)
    if not os.path.exists(export_dir_):
        os.mkdir(export_dir_)

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
    test_label_every_class = 0
    test_recall_every_class = 0
    test_precision_every_class = 0
    while timedelay < timedelay_num:
        if i >= len(train_data):
            i = 0
            # shuffle
            train_data_id = [i for i in range(len(train_data))]
            np.random.seed(8899)
            np.random.shuffle(train_data_id)
            train_data = [train_data[train_data_id[i]] for i in range(len(train_data))]

        features = train_data[i : i + batch_size]
        i += batch_size
        batches += 1
        model.train(sess, features[0])

        if batches % checkpoint_num == 0:
            train_loss, train_results = model.test(sess, features)
            train_label, train_precision, train_recall = evaluate_func.evaluate_func(smart_show.smart_show(train_results,
                                                                                                           multilabel=is_multilabel),
                                                                                     multilabel=is_multilabel)
            train_label = train_label[0]
            train_recall = train_recall[0]
            train_precision = train_precision[0]

            dev_loss, dev_results = model.test(sess, valid_data)
            dev_label, dev_precision, dev_recall = evaluate_func.evaluate_func(smart_show.smart_show(dev_results,
                                                                                                     multilabel=is_multilabel),
                                                                                     multilabel=is_multilabel)
            dev_label_every_class = dev_label[1]
            dev_precision_every_class = dev_precision[1]
            dev_recall_every_class = dev_recall[1]
            dev_label = dev_label[0]
            dev_recall = dev_recall[0]
            dev_precision = dev_precision[0]
            #sys.stdout.write('\nValid Label:' + str(dev_label_every_class))
            #sys.stdout.write('\nValid Recall:' + str(dev_recall_every_class))
            #sys.stdout.write('\nValid Precision:' + str( dev_precision_every_class))

            sys.stdout.write('\nTest Label:' + str(test_label_every_class))
            sys.stdout.write('\nTest Recall:' + str(test_recall_every_class))
            sys.stdout.write('\nTest Precision:' + str(test_precision_every_class))

            if best_loss > dev_loss or best_label < dev_label:
                timedelay = 0
                test_loss, test_results = model.test(sess, test_data)
                test_label, test_precision, test_recall = evaluate_func.evaluate_func(smart_show.smart_show(test_results,
                                                                                                            multilabel=is_multilabel),
                                                                                   multilabel=is_multilabel)
                test_label_every_class = test_label[1]
                test_recall_every_class = test_recall[1]
                test_precision_every_class = test_precision[1]

                test_label = test_label[0]
                test_recall = test_recall[0]
                test_precision = test_precision[0]

                # if os.path.exists(root_path + '/all_in_one/demo/exported_models/' + mission + '/' + str(1)):
                #     shutil.rmtree(root_path + '/all_in_one/demo/exported_models/' + mission + '/' + str(1))
                # export_func.export(model, sess, signature_name=mission, version=1)

                saver.save(sess, export_dir_ + 'test_model',global_step=batches)

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

            sys.stdout.write('\nBatches: %d' % batches)
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

            # sys.stdout.write('\n\n')

            start_time = time.time()

    print('\nModel saved at {}'.format(export_dir_))
    input_file = os.path.join(root_path, 'focus/Data/dc_labeled/labeled-Focus4Project-189-2018.01.09-test.json')
    method = 2
    watch_class = 2
    graph_path = 'cnn_model_hierarchical_supervision'
    graph_name = 'model_cnn_hierarchical_supervision'
    compare_fn(input_file, method, watch_class, model_name, graph_path, graph_name)


def main(argv):
    config_file = argv[1]
    round_num = argv[2]
    config = config_utils.Config()(root_path + '/all_in_one/config/' + mission + '/' + config_file)['model_parameter']
    config["round_num"] = round_num
    accs_label = []
    recalls_label = []
    precisions_label = []
    every_class_all = []
    path_prefix = root_path + '/all_in_one/data/data_train/' + mission_data + '/'

    train_data_path = os.path.join(path_prefix, 'data_train.json')
    valid_data_path = os.path.join(path_prefix, 'data_valid.json')
    test_data_path = os.path.join(path_prefix, 'data_test.json')

    checkpoint_dir = os.path.join(root_path, 'all_in_one/demo/exported_models/'
                                  'focus_hierarchical_supervision/'
                                  'batch_size_1-norm_lim_3.0-grad_lim_5.0-filter_num_300-lossweight_0_0_1/')
    #train(train_data_path, valid_data_path, test_data_path, path_prefix, load_model=checkpoint_dir)
    train(train_data_path, valid_data_path, test_data_path, path_prefix, config,
          load_model=None)


if __name__ == '__main__':
    tf.app.run()
