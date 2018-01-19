import os
import sys
import tensorflow as tf

root_path = sys.path[0]


def export(model, sess, signature_name, export_path=root_path + '/all_in_one/demo/exported_models/', version=1):
    # export path
    export_path = os.path.join(os.path.realpath(export_path), signature_name, str(version))
    print('Exporting trained model to {} ...'.format(export_path))

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # Build the signature_def_map.
    classification_w = tf.saved_model.utils.build_tensor_info(model.w)
    # classification_is_training = tf.saved_model.utils.build_tensor_info(model.is_training)
    classification_dropout_keep_prob_mlp = tf.saved_model.utils.build_tensor_info(
        model.dropout_keep_prob_mlp)
    # score
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(model.y)

    classification_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_w},
        outputs={
            tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
            classification_outputs_scores
        },
        method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)  # 'tensorflow/serving/classify'

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input_plh': classification_w, 'dropout_keep_prob_mlp':
                classification_dropout_keep_prob_mlp,
                # 'is_training': classification_is_training
                },
        outputs={'scores': classification_outputs_scores},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)  # 'tensorflow/serving/predict'
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            signature_name: prediction_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature,
        })
    builder.save()
