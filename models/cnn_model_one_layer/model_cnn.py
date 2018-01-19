import tensorflow as tf
import numpy as np

np.random.seed(8899)
tf.set_random_seed(8899)

from all_in_one.models.cnn_model_one_layer import utilizer
from all_in_one.layers.common_layers import layer_postprocess

class Model:
    def __init__(self, label_class, maxlen, W_embedding, label_dict, y_distribution, config, multilabel=False):
        self.dropout_mlp = float(config['dropout_mlp'])
        self.norm_lim = float(config['norm_lim'])
        self.grad_lim = float(config['grad_lim'])
        self.filter_num = int(config['filter_num'])
        self.filter_lengths = [int(fl) for fl in config['filter_lengths'].split()]
        self.label_class = label_class
        self.maxlen = maxlen
        self.label_dict = label_dict
        self.y_distribution = y_distribution
        self.multilabel = multilabel

        hparams = tf.contrib.training.HParams()
        hparams.add_hparam('layer_preprocess_sequence', 'none')
        hparams.add_hparam('layer_postprocess_sequence', 'none')
        hparams.add_hparam('layer_prepostprocess_dropout', self.dropout_mlp)
        hparams.add_hparam('norm_type', 'batch')
        hparams.add_hparam('hidden_size', self.filter_num)
        hparams.add_hparam('norm_epsilon', 1e-6)

        # input
        self.w = tf.placeholder(tf.int32, [None, None], name='input_w')
        self.sl = tf.placeholder(tf.int32, [None], name='input_sl')
        self.y_ = tf.placeholder(tf.float32, [None, None], name='input_y')
        self.dropout_keep_prob_mlp = tf.placeholder(tf.float32, name='dropout_keep_prob_mlp')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        def mlp_weight_variable(shape):
            initial = tf.random_normal(shape=shape, stddev=0.01)
            mlp_W = tf.Variable(initial, name='mlp_W')
            return tf.clip_by_norm(mlp_W, self.norm_lim)

        def mlp_bias_variable(shape):
            initial = tf.zeros(shape=shape)
            mlp_b = tf.Variable(initial, name='mlp_b')
            return mlp_b

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

        # word embedding
        with tf.device('/cpu:0'), tf.name_scope('word_embedding'):
            word_embedding_table = tf.Variable(np.asarray(W_embedding), name='word_embedding_table')
            word_embedded = tf.nn.embedding_lookup(word_embedding_table, self.w)

        # word CNN
        with tf.name_scope('CNN'):
            self.cnn_size = self.filter_num * len(self.filter_lengths)

            conv_Ws = []
            conv_bs = []
            for i in self.filter_lengths:
                filter_shape = [i,
                                W_embedding.shape[1],
                                self.filter_num]
                conv_W = conv_weight_variable(filter_shape)
                conv_b = conv_bias_variable([self.filter_num])
                conv_Ws.append(conv_W)
                conv_bs.append(conv_b)

            cnn_outputs = []
            for i in range(len(self.filter_lengths)):
                conv = conv1d(word_embedded, conv_Ws[i], conv_bs[i])
                cnn_outputs.append(conv)

            cnn_outputs_concat = tf.concat(cnn_outputs, 2)
            ## concat and layer postprocess
            #cnn_outputs_concat = layer_postprocess(
            #    None,
            #    tf.concat(cnn_outputs, 2),
            #    hparams,
            #    is_training=self.is_training)
            cnn_outputs_concat_act = tf.nn.relu(cnn_outputs_concat, name='relu')

            # predict label
            self.label_hidden_state = tf.reduce_max(cnn_outputs_concat_act, axis=1)
            self.label_hidden_state = tf.nn.dropout(self.label_hidden_state, self.dropout_keep_prob_mlp) * (1 + self.dropout_mlp - self.dropout_keep_prob_mlp)

            # one layer perceptron
            #label_W = mlp_weight_variable([self.cnn_size, self.label_class])
            #label_b = mlp_bias_variable([self.label_class])
            #h_outputs = tf.matmul(self.label_hidden_state, label_W) + label_b

            # two layer perceptron
            with tf.name_scope('fc_1'):
                hidden_dim = 100
                label_W1 = mlp_weight_variable([self.cnn_size, hidden_dim])
                label_b1 = mlp_bias_variable([hidden_dim])
                hidden = tf.matmul(self.label_hidden_state, label_W1) + label_b1
                hidden = tf.nn.relu(hidden, name='relu_1')

            with tf.name_scope('fc_2'):
                label_W2 = mlp_weight_variable([hidden_dim, self.label_class])
                label_b2 = mlp_bias_variable([self.label_class])
                h_outputs = tf.matmul(hidden, label_W2) + label_b2

            if self.multilabel:
                self.y = tf.nn.sigmoid(h_outputs)
            else:
                self.y = tf.nn.softmax(h_outputs)
            self.y = tf.clip_by_value(self.y, clip_value_min=1e-6, clip_value_max=1.0 - 1e-6)
            self.loss_label = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y) * self.y_distribution
                                                            + (1 - self.y_) * tf.log(1 - self.y), reduction_indices=[1]))

            if self.multilabel:
                self.prediction = tf.round(self.y)
            else:
                self.prediction = tf.reduce_max(self.y, keep_dims=True, axis=1)
                self.prediction = tf.cast(tf.equal(self.prediction, self.y), tf.float32)

            self.loss = self.loss_label

            # loss
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-08)
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [((tf.clip_by_norm(grad, self.grad_lim)), var) for grad, var in gvs]
            self.train_step = optimizer.apply_gradients(capped_gvs)
            # self.train_step = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-08).minimize(self.loss)

            # tvars = tf.trainable_variables()
            #
            # for var in tvars:
            #     print(var.name)
            # exit()

            # tf.global_variables_initializer().run()
            # w_batch = np.random.randint(2, 100, [2, self.maxlen])
            # sl_batch = [self.maxlen] * 2
            # y_batch = np.random.binomial(1, 0.5, [2, self.label_class])
            # test = self.loss.eval({self.w: w_batch,
            #                                 self.sl: sl_batch,
            #                                 self.y_: y_batch,
            #                                 self.dropout_keep_prob_mlp: 1.0})
            # print(test)
            # print(np.asarray(test).shape)
            # exit()

    def train(self, w, sl, y):
        w_batch = w
        sl_batch = sl
        y_batch = y

        w_batch = utilizer.pad_list(w_batch.tolist())

        self.train_step.run({self.w: w_batch,
                            self.sl: sl_batch,
                            self.y_: y_batch,
                            self.dropout_keep_prob_mlp: self.dropout_mlp,
                            self.is_training: True})

    def test(self, sess, w, sl, y, batch_size):
        loss = 0
        results = []
        i = 0
        while i < len(w):
            w_batch = w[i: i + batch_size]
            sl_batch = sl[i: i + batch_size]
            y_batch = y[i: i + batch_size]
            i += batch_size

            w_batch = utilizer.pad_list(w_batch.tolist())

            r = sess.run([self.loss, self.y],
                         feed_dict={self.w: w_batch,
                                    self.sl: sl_batch,
                                    self.y_: y_batch,
                                    self.dropout_keep_prob_mlp: 1.0,
                                    self.is_training: False})

            loss += len(w_batch) * r[0]
            logits = r[1]
            results.extend(zip(logits, y_batch.tolist()))

        try:
            loss /= len(w)
        except:
            loss = loss

        return loss, results

    def load_model(self, sess, checkpoint_dir, model_checkpoint_path=None):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        print('checkpoint_dir:', checkpoint_dir)
        if not model_checkpoint_path:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                model_checkpoint_path = ckpt.model_checkpoint_path
        print('model_checkpoint_path:', model_checkpoint_path)
        try:
            print('loading pretrained model')
            saver.restore(sess, model_checkpoint_path)
        except:
            raise('failed to load a model')

