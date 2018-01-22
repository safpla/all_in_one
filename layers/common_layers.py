# coding=utf-8
# __author__ == 'Xu Haowen'
import tensorflow as tf



def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""

def apply_norm(x, norm_type, depth, epsilon, is_training=True):
    if norm_type == 'layer':
        return layer_norm(x, filters=depth, epsilon=epsilon)
    if norm_type == 'batch':
        return tf.layers.batch_normalization(x, training=is_training,
                                             name='batch_norm')
    if norm_type == 'none':
        return x
    raise ValueError("Parameter normalizaer_fn must be one of: 'layer', 'batch'"
                     ", 'none'.")

def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         dropout_rate,
                         norm_type,
                         depth,
                         epsilon,
                         default_name,
                         is_training=True,
                         name=None):
    """Apply a sequence of functions to the input or output of a layer.

    The sequence is specified as a string which may contain the following
    characters:
      a: add previous_value
      n: apply normalization
      d: apply dropout

    For example, if sequence=="dna", then the output is
      previous_value + normalize(dropout(x))

    Args:
      previous_value: A Tensor, to be added as a residual connection ('a')
      x: A Tensor to be transformed.
      sequence: a string.
      dropout_rate: a float
      norm_type: a string (see apply_norm())
      depth: an integer (size of last dimension of x).
      epsilon: a float (parameter for normalization)
      default_name: a string
      name: a string

    Returns:
      a Tensor
    """
    with tf.variable_scope(name, default_name=default_name):
        if sequence == "none":
            return x
        for c in sequence:
            if c == "a":
                x += previous_value
            elif c == "n":
                x = apply_norm(x, norm_type, depth, epsilon, is_training)
            else:
                assert c == "d", ("Unknown sequence step %s" % c)
                x = tf.layers.dropout(x, 1.0 - dropout_rate, training=is_training)
        return x

def layer_preprocess(layer_input, hparams, sequence=None):
    """Apply layer preprocessing.

    See layer_prepostprocess() for details.

    A hyperparemeters object is passed for convenience.  The hyperparameters
    that may be used are:

      layer_preprocess_sequence
      layer_prepostprocess_dropout
      norm_type
      hidden_size
      norm_epsilon

    Args:
      layer_input: a Tensor
      hparams: a hyperparameters object.

    Returns:
      a Tensor
    """
    assert "a" not in hparams.layer_preprocess_sequence, (
        "No residual connections allowed in hparams.layer_preprocess_sequence")
    if not sequence:
        sequence = hparams.layer_postprocess_sequence

    return layer_prepostprocess(
        None,
        layer_input,
        sequence=sequence,
        dropout_rate=hparams.layer_prepostprocess_dropout,
        norm_type=hparams.norm_type,
        depth=hparams.hidden_size,
        epsilon=hparams.norm_epsilon,
        is_training=is_training,
        default_name="layer_prepostprocess")

def layer_postprocess(layer_input, layer_output, hparams, sequence=None,
                      is_training=True):
    """Apply layer postprocessing.

    See layer_prepostprocess() for details.

    A hyperparemeters object is passed for convenience.  The hyperparameters
    that may be used are:

      layer_postprocess_sequence
      layer_prepostprocess_dropout
      norm_type
      hidden_size
      norm_epsilon

    Args:
      layer_input: a Tensor
      layer_output: a Tensor
      hparams: a hyperparameters object.

    Returns:
      a Tensor
    """
    if not sequence:
        sequence = hparams.layer_postprocess_sequence

    return layer_prepostprocess(
        layer_input,
        layer_output,
        sequence=sequence,
        dropout_rate=hparams.layer_prepostprocess_dropout,
        norm_type=hparams.norm_type,
        depth=hparams.hidden_size,
        epsilon=hparams.norm_epsilon,
        is_training=is_training,
        default_name="layer_postprocess")

