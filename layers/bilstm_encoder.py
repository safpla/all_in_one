# coding=utf-8

import tensorflow as tf

def bilstm_encoder(inputs, sl, hidden_size=128, num_hidden_layers=1,
                   name='bilstm_encoder',
                   dropout=0.5,
                   is_training=True):
    """
    Run bi-LSTM encoder on inputs,
    inputs: [batch_size, time_steps, hidden_size]
    outputs: the output of bidirectional_dynamic_rnn
    """

    def dropout_lstm_cell():
        # DropoutWrapper: adding dropout to inputs and outputs of the given cell
        # MultiRNNCell: creating a RNN cell composed sequentially of a number of
        #               RNNCells
        # bidirectional_dynamic_rnn
        return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(hidden_size),
            input_keep_prob=1.0 - dropout * tf.to_float(is_training))

    with tf.variable_scope(name):
        # create cells
        cell_fw = [dropout_lstm_cell() for _ in range(num_hidden_layers)]
        cell_bw = [dropout_lstm_cell() for _ in range(num_hidden_layers)]
        # create RNN cells composed sequentially of a number of RNNCells
        cell_fw = tf.contrib.rnn.MultiRNNCell(cell_fw)
        cell_bw = tf.contrib.rnn.MultiRNNCell(cell_bw)

        return tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=sl,
            dtype=tf.float32,
            time_major=False)

def lstm_decoder_no_feedback(inputs, sl, hidden_size=128, num_hidden_layers=1,
               name='lstm',
               dropout=0.5,
               is_training=True):
    """
    LSTM layers,
    input: [batch_size, time_steps, hidden_size]
    """
    def dropout_lstm_cell():
        # DropoutWrapper: adding dropout to inputs and outputs of the given cell
        # MultiRNNCell: creating a RNN cell composed sequentially of a number of
        #               RNNCells
        # bidirectional_dynamic_rnn
        return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(hidden_size),
            input_keep_prob=1.0 - dropout * tf.to_float(is_training))

    with tf.variable_scope(name):
        # create cells
        cell = [dropout_lstm_cell() for _ in range(num_hidden_layers)]
        # create a RNN cell composed sequentially of a number of RNNCells
        cell = tf.contrib.rnn.MultiRNNCell(cell)

        return tf.nn.dynamic_rnn(cell=cell,
                                 inputs=inputs,
                                 dtype=tf.float32,
                                 sequence_length=sl,
                                 time_major=False)


