import tensorflow as tf
import numpy as np

np.random.seed(8899)
tf.set_random_seed(8899)

from all_in_one.models.cnn_model_one_layer_tag import utilizer


class Model:
    def __init__(self, label_class, maxlen, W_embedding, label_dict, y_distribution, tag_class, config, multilabel=False):
        self.dropout_mlp = float(config['dropout_mlp'])
        self.tag_dims = int(config['tag_dims'])
        self.norm_lim = float(config['norm_lim'])
        self.grad_lim = float(config['grad_lim'])
        self.filter_num = int(config['filter_num'])
        self.filter_lengths = [int(fl) for fl in config['filter_lengths'].split()]
        self.label_class = label_class
        self.maxlen = maxlen
        self.label_dict = label_dict
        self.y_distribution = y_distribution
        self.tag_class = tag_class
        self.multilabel = multilabel

        # input
        self.w = tf.placeholder(tf.int32, [None, None], name='input_w')
        self.t = tf.placeholder(tf.int32, [None, None, None], name='input_t')
        self.sl = tf.placeholder(tf.int32, [None], name='input_sl')
        self.y_ = tf.placeholder(tf.float32, [None, None], name='input_y')
        self.dropout_keep_prob_mlp = tf.placeholder(tf.float32, name='dropout_keep_prob_mlp')

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

        # tag embedding
        with tf.device('/cpu:0'), tf.name_scope('tag_embedding'):
            initial = np.random.uniform(-0.25, 0.25, [self.tag_class, self.tag_dims])
            tag_embedding_table = tf.Variable(initial, dtype=tf.float32, name='tag_embedding_table')
            tag_embedded = tf.nn.embedding_lookup(tag_embedding_table, self.t)
            tag_embedded = tf.reshape(tag_embedded, [tf.shape(tag_embedded)[0], tf.shape(tag_embedded)[1], -1])

        cnn_inputs = tf.concat([word_embedded, tag_embedded], axis=2)

        # word CNN
        with tf.name_scope('CNN'):
            self.cnn_size = self.filter_num * len(self.filter_lengths)

            conv_Ws = []
            conv_bs = []
            for i in self.filter_lengths:
                filter_shape = [i,
                                W_embedding.shape[1] + int(self.tag_class * self.tag_dims / 2),
                                self.filter_num]
                conv_W = conv_weight_variable(filter_shape)
                conv_b = conv_bias_variable([self.filter_num])
                conv_Ws.append(conv_W)
                conv_bs.append(conv_b)

            cnn_outputs = []
            for i in range(len(self.filter_lengths)):
                conv = conv1d(cnn_inputs, conv_Ws[i], conv_bs[i])
                cnn_outputs.append(conv)
            cnn_outputs_concat = tf.concat(cnn_outputs, 2)
            cnn_outputs_concat_act = tf.nn.relu(cnn_outputs_concat, name='relu')

            # predict label
            self.label_hidden_state = tf.reduce_max(cnn_outputs_concat_act, axis=1)
            self.label_hidden_state = tf.nn.dropout(self.label_hidden_state, self.dropout_keep_prob_mlp) * (1 + self.dropout_mlp - self.dropout_keep_prob_mlp)
            label_W = mlp_weight_variable([self.cnn_size, self.label_class])
            label_b = mlp_bias_variable([self.label_class])
            h_outputs = tf.matmul(self.label_hidden_state, label_W) + label_b
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
            # t_batch = np.random.binomial(1, 0.5, [2, self.maxlen, 3])
            # sl_batch = [self.maxlen] * 2
            # y_batch = np.random.binomial(1, 0.5, [2, self.label_class])
            # test = self.loss.eval({self.w: w_batch,
            #                        self.t: t_batch,
            #                        self.sl: sl_batch,
            #                        self.y_: y_batch,
            #                        self.dropout_keep_prob_mlp: 1.0})
            # print(test)
            # print(np.asarray(test).shape)
            # exit()

    def train(self, w, t, sl, y):
        w_batch = w
        t_batch = t
        sl_batch = sl
        y_batch = y

        w_batch = utilizer.pad_list(w_batch.tolist(), 0)
        t_batch = utilizer.pad_list(t_batch.tolist(), [2 * i for i in range(self.tag_class // 2)])

        self.train_step.run({self.w: w_batch,
                            self.t: t_batch,
                            self.sl: sl_batch,
                            self.y_: y_batch,
                            self.dropout_keep_prob_mlp: self.dropout_mlp})

    def test(self, sess, w, t, sl, y, batch_size):
        loss = 0
        results = []
        i = 0
        while i < len(w):
            w_batch = w[i: i + batch_size]
            t_batch = t[i: i + batch_size]
            sl_batch = sl[i: i + batch_size]
            y_batch = y[i: i + batch_size]
            i += batch_size

            w_batch = utilizer.pad_list(w_batch.tolist(), 0)
            t_batch = utilizer.pad_list(t_batch.tolist(), [2 * i for i in range(self.tag_class // 2)])

            r = sess.run([self.loss, self.y],
                         feed_dict={self.w: w_batch,
                                    self.t: t_batch,
                                    self.sl: sl_batch,
                                    self.y_: y_batch,
                                    self.dropout_keep_prob_mlp: 1.0})

            loss += len(w_batch) * r[0]
            logits = r[1]
            results.extend(zip(logits, y_batch.tolist()))

        loss /= len(w)

        return loss, results
