import tensorflow as tf
import numpy as np
import os, sys
root_path = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, root_path)

from all_in_one.layers.bilstm_encoder import bilstm_encoder, lstm_decoder_no_feedback
from all_in_one.models.bilstm_model_one_layer import utilizer

np.random.seed(8899)
tf.set_random_seed(8899)

class Model:
    def init_global_step(self):
        config = self.config
        # Global steps for asynchronous distributed training.
        with tf.device(str(config['device'])):
            self.global_step = tf.get_variable('global_step', [],
                                               initializer=tf.constant_initializer(0),
                                               trainable=False)

    def _create_placeholder(self):
        config = self.config
        self.input_plh = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name='input_plh')

        self.label_plh = tf.placeholder(
            dtype=tf.float32,
            shape=[None, None],
            name='label_plh')

        self.sl_plh = tf.placeholder(
            dtype=tf.int32,
            shape=[None],
            name='sl_plh')

        self.is_training = tf.placeholder(
            dtype=tf.bool,
            shape=[],
            name='is_training')

    def _inference_graph(self):
        config = self.config
        hidden_size = int(config['hidden_size'])
        num_hidden_layers = int(config['num_hidden_layers'])
        dropout_lstm = float(config['dropout_lstm'])
        num_classes = int(config['num_classes'])
        norm_lim = float(config['norm_lim'])
        grad_lim = float(config['grad_lim'])
        learning_rate = float(config['learning_rate'])
        rho = float(config['rho'])

        def fc_layer(inputs, num_classes):
            with tf.variable_scope("fc_layer"):
                w = tf.Variable(tf.random_normal(shape=[hidden_size, num_classes],
                                                 stddev=0.01),
                                name='fc_w')
                b = tf.Variable(tf.zeros(shape=[num_classes]),
                                name='fc_b')
                outputs = tf.matmul(inputs, w) + b
                return outputs


        # word embedding
        with tf.device('/cpu:0'), tf.name_scope('word_embedding'):
            word_embedding_table = tf.Variable(np.asarray(self.w_embedding),
                                              name='word_embedding_table')
            word_embedded = tf.nn.embedding_lookup(word_embedding_table,
                                                   self.input_plh)
        self.word_embedded = word_embedded
        print('embedding', word_embedded)
        with tf.variable_scope("lstm_seq2label"):
            # bi-LSTM encoder
            output, enc_state = bilstm_encoder(
                word_embedded,
                self.sl_plh,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                name='bilstm_encoder',
                dropout=dropout_lstm,
                is_training=self.is_training,
                )
            self.o_fw = output[0]
            self.o_bw = output[1]
            #h_fw = enc_state[0][0][1]
            #h_bw = enc_state[1][0][1]
            #self.h_fw = h_fw
            #self.h_bw = h_bw
            enc_state = tf.concat([self.o_fw, self.o_bw], axis=1)

            # LSTM decoder
            _, dec_state = lstm_decoder_no_feedback(
                enc_state,
                self.sl_plh,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                name='lstm_decoder',
                dropout=dropout_lstm,
                is_training=self.is_training,
            )

            dec_state = dec_state[0][1]
            self.dec_state = dec_state
            # classifier
            fc_output = fc_layer(dec_state, num_classes)
            self.fc_output = fc_output
            if self.multilabel:
                self.logits = tf.nn.sigmoid(fc_output)
            else:
                self.logits = tf.nn.softmax(fc_output)
            self.logits = tf.clip_by_value(self.logits, clip_value_min=1e-6,
                                           clip_value_max=1.0-1e-6)
            if self.multilabel:
                self.prediction = tf.round(self.logits)
            else:
                self.prediction = tf.reduce_max(self.logits, keep_dims=True, axis=1)
                self.prediction = tf.cast(tf.equal(self.predictioin, self.logits),
                                          tf.float32)

            # loss
            self.loss = tf.reduce_mean(
                -tf.reduce_sum(self.label_plh * tf.log(self.logits) * self.y_distribution
                               + (1 - self.label_plh) * tf.log(1 - self.logits),
                               reduction_indices=[1]))
            self.optimizer = tf.train.AdadeltaOptimizer(
                learning_rate=learning_rate,
                rho=rho,
                epsilon=1e-8)
            gvs = self.optimizer.compute_gradients(self.loss)
            capped_gvs = [((tf.clip_by_norm(grad, grad_lim)), var) for grad, var in gvs]
            self.train_step = self.optimizer.apply_gradients(capped_gvs)

    def __init__(self, w_embedding, y_distribution, config, multilabel=False,
                 model_name='bi-lstm'):
        self.config = config
        self.w_embedding = w_embedding
        self.multilabel = multilabel
        self.y_distribution = y_distribution

        with tf.variable_scope(model_name):
            self.init_global_step()
            self._create_placeholder()
            self._inference_graph()

    def train(self, sess, w, sl, y):
        w_batch = w
        sl_batch = sl
        y_batch = y
        w_batch, sl_batch = utilizer.remove_padding(w_batch.tolist())
        w_batch = utilizer.pad_list(w_batch.tolist())
        #print('w_batch', w_batch)
        #print('sl_plh', sl_batch)
        print('label:\n')
        y_batch_print = y_batch.tolist()
        for line in y_batch_print:
            print(line)

        checkout = sess.run([self.o_fw, self.o_bw, self.dec_state, self.logits], feed_dict={
            self.input_plh: w_batch,
            self.sl_plh: sl_batch,
            self.label_plh: y_batch,
            self.is_training: True})
        #print('o_fw: ', checkout[0])
        #print(len(checkout[0][0]), len(checkout[0][1]))
        #print('o_bw: ', checkout[1])
        #print(len(checkout[1][0]), len(checkout[1][1]))
        #print('h_fw: ', checkout[2])
        #print('h_bw: ', checkout[3])
        #print('dec_state: ', checkout[4])
        print('logits: \n')
        logits_print = checkout[3].tolist()
        for line in logits_print:
            print(line)

        self.train_step.run({self.input_plh: w_batch,
                            self.sl_plh: sl_batch,
                            self.label_plh: y_batch,
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

            w_batch, sl_batch = utilizer.remove_padding(w_batch.tolist())
            w_batch = utilizer.pad_list(w_batch.tolist())

            r = sess.run([self.loss, self.logits, self.prediction],
                         feed_dict={self.input_plh: w_batch,
                                    self.sl_plh: sl_batch,
                                    self.label_plh: y_batch,
                                    self.is_training: False})
            loss += len(w_batch) * r[0]
            logits = r[1]
            prediction = r[2]
            results.extend(zip(logits, y_batch.tolist()))
        loss /= len(w)
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
