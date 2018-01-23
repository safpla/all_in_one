import tensorflow as tf
import numpy as np
import os, sys
root_path = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, root_path)

from all_in_one.layers.bilstm_encoder import bilstm_encoder, lstm_decoder_no_feedback
from all_in_one.models.bilstm_model_one_layer import utilizer
from all_in_one.layers.common_layers import layer_preprocess, layer_postprocess

np.random.seed(8899)
tf.set_random_seed(8899)

def select_label(in_label, select_id):
    if in_label.ndim == 2:
        out_label = in_label[:, select_id]
    else:
        out_label = in_label[select_id]
    return out_label


class Model:
    def init_global_step(self):
        config = self.config
        # Global steps for asynchronous distributed training.
        with tf.device(str(config['device'])):
            self.global_step = tf.get_variable('global_step', [],
                                               initializer=tf.constant_initializer(0),
                                               trainable=False)

    def _create_placeholder(self):
        # batch_size should be 1 in this version

        config = self.config
        iClass = int(config['iClass'])
        if iClass == -1:
            num_classes = int(config['num_classes'])
        else:
            num_classes = 1
        self.dfdt_input_plh = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None], # [sentence, word]
            name='dfdt_input_plh')

        self.dfdt_label_plh = tf.placeholder(
            dtype=tf.float32,
            shape=[None, num_classes], # [(sentence, class]
            name='dfdt_label_plh')

        self.dfdt_sl_plh = tf.placeholder(
            dtype=tf.int32,
            shape=[None], # [sentence]
            name='dfdt_sl_plh')

        self.court_input_plh = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None], # [sentence, word]
            name='court_input_plh')

        self.court_label_plh = tf.placeholder(
            dtype=tf.float32,
            shape=[None, num_classes], # [sentence, class]
            name='court_label_plh')

        self.court_sl_plh = tf.placeholder(
            dtype=tf.int32,
            shape=[None], # [sentence]
            name='court_sl_plh')

        self.docu_label_plh = tf.placeholder(
            dtype=tf.float32,
            shape=[num_classes], # [class]
            name='docu_label_plh')

        # self.dfdt_input_plh = tf.placeholder(
        #     dtype=tf.int32,
        #     shape=[None, None, None], # [document(batch), sentence, word]
        #     name='dfdt_input_plh')

        # self.dfdt_label_plh = tf.placeholder(
        #     dtype=tf.float32,
        #     shape=[None, None, None], # [document(batch), sentence, class]
        #     name='dfdt_label_plh')

        # self.dfdt_sl_plh = tf.placeholder(
        #     dtype=tf.int32,
        #     shape=[None, None], # [document(batch), sentence]
        #     name='dfdt_sl_plh')

        # self.court_input_plh = tf.placeholder(
        #     dtype=tf.int32,
        #     shape[None, None, None], # [document(batch), sentence, word]
        #     name='court_input_plh')

        # self.court_label_plh = tf.placeholder(
        #     dtype=tf.float32,
        #     shape=[None, None, None], # [document(batch), sentence, class]
        #     name='court_label_plh')

        # self.court_sl_plh = tf.placeholder(
        #     dtype=tf.int32,
        #     shape=[None, None], # [document(batch), sentence]
        #     name='court_sl_plh')

        # self.docu_label_plh = tf.placeholder(
        #     dtype=tf.float32,
        #     shape=[None, None], # [document(batch), class]
        #     name='docu_label_plh')

        self.is_training = tf.placeholder(
            dtype=tf.bool,
            shape=[],
            name='is_training')

    def _inference_graph(self):
        config = self.config
        norm_lim = float(config['norm_lim'])
        grad_lim = float(config['grad_lim'])
        learning_rate = float(config['learning_rate'])
        rho = float(config['rho'])
        embedding_dim = int(config['embedding_dim'])
        num_classes = int(config['num_classes'])
        filter_num = int(config['filter_num'])
        filter_lengths = [int(fl) for fl in config['filter_lengths'].split()]
        loss_weights = [float(lw) for lw in config['loss_weights'].split()]
        dfdt_only = [int(do) for do in config['dfdt_only'].split()]
        sepa_conv = bool(int(config['sepa_conv']))

        hparams = tf.contrib.training.HParams()
        hparams.add_hparam('layer_preprocess_sequence',
                            config['layer_preprocess_sequence'])
        hparams.add_hparam('layer_postprocess_sequence',
                            config['layer_postprocess_sequence'])
        hparams.add_hparam('layer_prepostprocess_dropout',
                            float(config['layer_prepostprocess_dropout']))
        hparams.add_hparam('norm_type', config['norm_type'])
        hparams.add_hparam('hidden_size', int(config['hidden_size']))
        hparams.add_hparam('norm_epsilon', float(config['norm_epsilon']))

        def conv_weight_variable(shape):
            initial = tf.random_uniform(shape, maxval=0.01)
            conv_W = tf.Variable(initial, name='conv_W', dtype=tf.float32)
            return conv_W

        def conv_bias_variable(shape):
            initial = tf.zeros(shape=shape)
            conv_b = tf.Variable(initial, name='conv_b', dtype=tf.float32)
            return conv_b

        def conv1d(x, conv_W, conv_b):
            conv = tf.nn.conv1d(x,
                                conv_W,
                                stride=1,
                                padding='SAME',
                                name='conv')
            conv = tf.nn.bias_add(conv, conv_b)
            return conv

        def mlp_weight_variable(shape):
            initial = tf.random_normal(shape=shape, stddev=0.01)
            mlp_W = tf.Variable(initial, name='mlp_W')
            return tf.clip_by_norm(mlp_W, norm_lim)

        def mlp_bias_variable(shape):
            initial = tf.zeros(shape=shape)
            mlp_b = tf.Variable(initial, name='mlp_b')
            return mlp_b

        def cnn_one_layer(inputs, label, sl):
            with tf.name_scope('CNN'):
                cnn_size = filter_num * len(filter_lengths)
                conv_Ws = []
                conv_bs = []
                for i in filter_lengths:
                    filter_shape = [i, embedding_dim, filter_num]
                    conv_W = conv_weight_variable(filter_shape)
                    conv_b = conv_bias_variable([filter_num])
                    conv_Ws.append(conv_W)
                    conv_bs.append(conv_b)

                cnn_outputs = []
                for i in range(len(filter_lengths)):
                    conv = conv1d(inputs, conv_Ws[i], conv_bs[i])
                    cnn_outputs.append(conv)

                # concat and layer postprocess
                cnn_outputs_concat = layer_postprocess(
                    None,
                    tf.concat(cnn_outputs, 2),
                    hparams,
                    is_training=self.is_training)

                cnn_outputs_concat_act = tf.nn.relu(cnn_outputs_concat, name='relu')

                # max pooling over kernels and dropout
                label_hidden_state = layer_postprocess(
                    None,
                    tf.reduce_max(cnn_outputs_concat_act, axis=1),
                    hparams,
                    sequence='d',
                    is_training=self.is_training)

                # one layer mlp
                #label_W = mlp_weight_variable([cnn_size, num_classes])
                #label_b = mlp_bias_variable([num_classes])
                #h_outputs = tf.matmul(label_hidden_state, label_W) + label_b

                # two layer mlp
                with tf.name_scope('fc_1'):
                    hidden_dim = int(config['hidden_dim'])
                    label_W1 = mlp_weight_variable([cnn_size, hidden_dim])
                    label_b1= mlp_bias_variable([hidden_dim])
                    hidden = tf.matmul(label_hidden_state, label_W1) + label_b1
                    hidden = tf.nn.relu(hidden, name='relu_1')
                with tf.name_scope('fc_2'):
                    label_W2 = mlp_weight_variable([hidden_dim, num_classes])
                    label_b2 = mlp_bias_variable([num_classes])
                    h_outputs = tf.matmul(hidden, label_W2) + label_b2

                if self.multilabel:
                    logits = tf.nn.sigmoid(h_outputs)
                else:
                    logits = tf.nn.softmax(h_outputs)
                logits = tf.clip_by_value(logits, clip_value_min=1e-6, clip_value_max=1.0 - 1e-6)
                loss = tf.reduce_mean(-tf.reduce_sum(
                    label * tf.log(logits) * self.y_distribution[0]
                    + (1 - label) * tf.log(1 - logits) * self.y_distribution[1],
                    reduction_indices=[1]))

                # max pooling over sentences
                logits_para = tf.reduce_max(logits, axis=0)

                return logits_para, logits, loss

        def cnn_one_layer_sepa_conv(inputs, label, sl):
            with tf.name_scope('CNN'):
                cnn_size = filter_num * len(filter_lengths)
                h_outputs = []
                for iClass in range(num_classes):
                    with tf.name_scope('class_{}'.format(iClass)):
                        conv_Ws = []
                        conv_bs = []
                        for i in filter_lengths:
                            filter_shape = [i, embedding_dim, filter_num]
                            conv_W = conv_weight_variable(filter_shape)
                            conv_b = conv_bias_variable([filter_num])
                            conv_Ws.append(conv_W)
                            conv_bs.append(conv_b)

                        cnn_outputs = []
                        for i in range(len(filter_lengths)):
                            conv = conv1d(inputs, conv_Ws[i], conv_bs[i])
                            cnn_outputs.append(conv)

                        cnn_outputs_concat = tf.concat(cnn_outputs, 2)
                        # concat and layer postprocess
                        #cnn_outputs_concat = layer_postprocess(
                        #    None,
                        #    tf.concat(cnn_outputs, 2),
                        #    hparams,
                        #    is_training=self.is_training)
                        cnn_outputs_concat_act = tf.nn.relu(cnn_outputs_concat, name='relu')

                        label_hidden_state = tf.reduce_max(cnn_outputs_concat_act, axis=1)
                        label_hidden_state = tf.layers.dropout(label_hidden_state,
                                                               rate=0.5,
                                                               training=self.is_training)
                        # max pooling over kernels and dropout
                        #label_hidden_state = layer_postprocess(
                        #    None,
                        #    tf.reduce_max(cnn_outputs_concat_act, axis=1),
                        #    hparams,
                        #    sequence='d',
                        #    is_training=self.is_training)

                        # one layer mlp
                        #label_W = mlp_weight_variable([cnn_size, 1])
                        #label_b = mlp_bias_variable([1])
                        #h_outputs_oneclass = tf.matmul(label_hidden_state, label_W) + label_b

                        # two layer mlp
                        with tf.name_scope('fc_1'):
                            hidden_dim = int(config['hidden_dim'])
                            label_W1 = mlp_weight_variable([cnn_size, hidden_dim])
                            label_b1= mlp_bias_variable([hidden_dim])
                            hidden = tf.matmul(label_hidden_state, label_W1) + label_b1
                            hidden = tf.nn.relu(hidden, name='relu_1')
                        with tf.name_scope('fc_2'):
                            label_W2 = mlp_weight_variable([hidden_dim, 1])
                            label_b2 = mlp_bias_variable([1])
                            h_outputs_oneclass = tf.matmul(hidden, label_W2) + label_b2

                        h_outputs.append(h_outputs_oneclass)

                h_outputs_concat = tf.concat(h_outputs, 1)

                if self.multilabel:
                    logits = tf.nn.sigmoid(h_outputs_concat)
                else:
                    logits = tf.nn.softmax(h_outputs_concat)
                logits = tf.clip_by_value(logits, clip_value_min=1e-6, clip_value_max=1.0 - 1e-6)
                loss = tf.reduce_mean(-tf.reduce_sum(
                    label * tf.log(logits) * self.y_distribution[0]
                    + (1 - label) * tf.log(1 - logits) * self.y_distribution[1],
                    reduction_indices=[1]))

                # max pooling over sentences
                logits_para = tf.reduce_max(logits, axis=0)

                return logits_para, logits, loss


        # word embedding
        with tf.device('/cpu:0'), tf.name_scope('word_embedding'):
            word_embedding_table = tf.Variable(np.asarray(self.w_embedding),
                                              name='word_embedding_table')
            dfdt_input_embedded = tf.nn.embedding_lookup(word_embedding_table,
                                                self.dfdt_input_plh,
                                                name='dfdt_embedded')
            court_input_embedded = tf.nn.embedding_lookup(word_embedding_table,
                                                 self.court_input_plh,
                                                 name='court_embedded')

        with tf.variable_scope("dfdt"):
            if sepa_conv:
                dfdt_logits, dfdt_logits_sents, dfdt_loss = cnn_one_layer_sepa_conv(
                    dfdt_input_embedded,
                    self.dfdt_label_plh,
                    self.dfdt_sl_plh)
            else:
                dfdt_logits, dfdt_logits_sents, dfdt_loss = cnn_one_layer(
                    dfdt_input_embedded,
                    self.dfdt_label_plh,
                    self.dfdt_sl_plh)

            self.dfdt_logits = dfdt_logits
            self.dfdt_logits_sents = dfdt_logits_sents
            self.dfdt_loss = dfdt_loss

        with tf.variable_scope("court"):
            if sepa_conv:
                court_logits, court_logits_sents, court_loss = cnn_one_layer_sepa_conv(
                    court_input_embedded,
                    self.court_label_plh,
                    self.court_sl_plh)
            else:
                court_logits, court_logits_sents, court_loss = cnn_one_layer(
                    court_input_embedded,
                    self.court_label_plh,
                    self.court_sl_plh)
            self.court_logits = court_logits
            self.court_logits_sents = court_logits_sents
            self.court_loss = court_loss

        with tf.variable_scope("para_reasoning"):
            min_mask_init = [0 for _ in range(num_classes)]
            max_mask_init = [1 for _ in range(num_classes)]
            for i in dfdt_only:
                min_mask_init[i] = 1
                max_mask_init[i] = 0

            min_mask = tf.constant(min_mask_init, dtype=dfdt_logits.dtype)
            max_mask = tf.constant(max_mask_init, dtype=dfdt_logits.dtype)

            # min logic doesn't work well
            #docu_logits = (tf.reduce_min(tf.stack([dfdt_logits * min_mask, court_logits * min_mask]), axis=0) +
            #               tf.reduce_max(tf.stack([dfdt_logits * max_mask, court_logits * max_mask]), axis=0))

            # use dfdt only for min logic
            docu_logits = (dfdt_logits * min_mask +
                           tf.reduce_max(tf.stack([dfdt_logits * max_mask, court_logits * max_mask]), axis=0))

            docu_logits = tf.clip_by_value(docu_logits, clip_value_min=1e-6, clip_value_max=1.0-1e-6)
            docu_loss = -tf.reduce_sum(self.docu_label_plh * tf.log(docu_logits) * self.y_distribution[0]
                                       + (1 - self.docu_label_plh) * tf.log(1 - docu_logits) * self.y_distribution[1],
                                       reduction_indices=[0])
            self.docu_logits = docu_logits
            self.docu_loss = docu_loss

        if self.multilabel:
            self.dfdt_prediction = tf.round(dfdt_logits)
            self.court_prediction = tf.round(court_logits)
            self.docu_prediction = tf.round(docu_logits)
        else:
            raise NotImplementedError

        # loss
        self.loss = (dfdt_loss * loss_weights[0] +
                     court_loss * loss_weights[1] +
                     docu_loss * loss_weights[2])

        self.optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=learning_rate,
            rho=rho,
            epsilon=1e-8)
        gvs = self.optimizer.compute_gradients(self.loss)
        capped_gvs = [((tf.clip_by_norm(grad, grad_lim)), var) for grad, var in gvs]
        self.train_step = self.optimizer.apply_gradients(capped_gvs)

    def __init__(self, w_embedding, y_distribution, config, multilabel=False,
                 model_name='cnn_hs'):
        self.config = config
        self.w_embedding = w_embedding
        self.multilabel = multilabel
        self.y_distribution = y_distribution
        self.iClass = int(config['iClass'])

        with tf.variable_scope(model_name):
            self.init_global_step()
            self._create_placeholder()
            self._inference_graph()

    def train(self, sess, features):
        dfdt_input = np.asarray(features['dfdt_input'])
        dfdt_input = utilizer.pad_list(dfdt_input.tolist())
        dfdt_label = np.asarray(features['dfdt_label'])
        dfdt_sl = np.asarray(features['dfdt_sl'])

        court_input = np.asarray(features['court_input'])
        court_input = utilizer.pad_list(court_input.tolist())
        court_label = np.asarray(features['court_label'])
        court_sl = np.asarray(features['court_sl'])

        docu_label = np.asarray(features['docu_label'])

        if self.iClass > -1:
            dfdt_label = select_label(dfdt_label, [self.iClass])
            court_label = select_label(court_label, [self.iClass])
            docu_label = select_label(docu_label, [self.iClass])

        feed_dict = {self.court_input_plh: court_input,
                     self.court_label_plh: court_label,
                     self.court_sl_plh: court_sl,
                     self.dfdt_input_plh: dfdt_input,
                     self.dfdt_label_plh: dfdt_label,
                     self.dfdt_sl_plh: dfdt_sl,
                     self.docu_label_plh: docu_label,
                     self.is_training: True}

        self.train_step.run(feed_dict)

        # checkout = [self.dfdt_logits, self.dfdt_loss,
        #             self.court_logits, self.court_loss,
        #             self.docu_logits, self.docu_loss,
        #             self.loss]
        # r = sess.run(checkout, feed_dict=feed_dict)
        # print('dfdt_logits:', r[0])
        # print('court_logits:', r[2])
        # print('docu_logits:', r[4])

    def test(self, sess, features_list, return_sents_result=False):
        loss = 0
        results = []
        dfdt_results = []
        court_results = []
        i = 0

        while i < len(features_list):
            features = features_list[i]
            i += 1

            dfdt_input = np.asarray(features['dfdt_input'])
            dfdt_input = utilizer.pad_list(dfdt_input.tolist())
            dfdt_label = np.asarray(features['dfdt_label'])
            dfdt_sl = np.asarray(features['dfdt_sl'])

            court_input = np.asarray(features['court_input'])
            court_input = utilizer.pad_list(court_input.tolist())
            court_label = np.asarray(features['court_label'])
            court_sl = np.asarray(features['court_sl'])

            docu_label = np.asarray(features['docu_label'])

            if self.iClass > -1:
                dfdt_label = select_label(dfdt_label, [self.iClass])
                court_label = select_label(court_label, [self.iClass])
                docu_label = select_label(docu_label, [self.iClass])

            feed_dict = {self.dfdt_input_plh: dfdt_input,
                         self.dfdt_label_plh: dfdt_label,
                         self.dfdt_sl_plh: dfdt_sl,
                         self.court_input_plh: court_input,
                         self.court_label_plh: court_label,
                         self.court_sl_plh: court_sl,
                         self.docu_label_plh: docu_label,
                         self.is_training: False}

            checkout = [self.dfdt_logits_sents, self.dfdt_loss,
                        self.court_logits_sents, self.court_loss,
                        self.docu_logits, self.docu_loss,
                        self.loss]
            r = sess.run(checkout, feed_dict=feed_dict)

            dfdt_logits_sents = r[0]
            dfdt_loss = r[1]
            court_logits_sents = r[2]
            court_loss = r[3]
            docu_logits = r[4]
            docu_loss = r[5]
            loss += r[6]
            # print('dfdt_logits: ', dfdt_logits)
            # print('dfdt_label: ', max(dfdt_label))
            # print('court_logits: ', court_logits)
            # print('court_label: ', max(court_label))
            # print('docu_logits: ', docu_logits)
            # print('docu_label: ', docu_label)
            results.append([docu_logits, docu_label.tolist()])
            dfdt_results.append(dfdt_logits_sents)
            court_results.append(court_logits_sents)
        loss /= len(features_list)
        if return_sents_result:
            return loss, results, dfdt_results, court_results
        else:
            return loss, results

    def load_model(self, sess, checkpoint_dir, model_checkpoint_path=None):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print(checkpoint_dir)
        if not model_checkpoint_path:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                model_checkpoint_path = ckpt.model_checkpoint_path
        print(model_checkpoint_path)
        try:
            print('loading pretrained model')
            saver.restore(sess, model_checkpoint_path)
        except:
            raise('failed to load a model')


if __name__ == '__main__':
    t = [float(x)/10 for x in range(200)]
    import math
    features = []
    labels = []
    a = [4 for _ in range(50)]
    b = [6 for _ in range(50)]
    sl = []
    for _ in range(10):
        features.append(a)
        labels.append([0, 1])
        sl.append(len(a))
        features.append(b)
        labels.append([1, 0])
        sl.append(len(b))

    tf.reset_default_graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    model = Model(W_embedding, config, multilabel=is_multilabel)

    sess.run(tf.global_variables_initializer())

    epoch_num = 100
    for epoch in epoch_num:
        model.train(sess, features, sl_batch, y_batch)
