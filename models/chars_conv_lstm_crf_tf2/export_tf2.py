"""Export model as a saved_model

Original Author: Guillaume Genthial
Modified by: Ivan Provalov
Project: Tensorflow 2 Upgraded chars_conv_lstm_crf (derived from chars_conv_lstm_crf)
Original https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/
License: Apache License

"""


from pathlib import Path
import json

import tensorflow as tf

from main_tf2 import model_fn

DATADIR = '../../data/example'
PARAMS = './results/params.json'
MODELDIR = './results/model'


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    words = tf.keras.Input(dtype=tf.string, shape=(None,), name='words')
    nwords = tf.keras.Input(dtype=tf.int32, shape=(), name='nwords')
    chars = tf.keras.Input(dtype=tf.string, shape=(None, None),
                           name='chars')
    nchars = tf.keras.Input(dtype=tf.int32, shape=(None,),
                            name='nchars')
    receiver_tensors = {'words': words, 'nwords': nwords,
                        'chars': chars, 'nchars': nchars}
    features = {'words': words, 'nwords': nwords,
                'chars': chars, 'nchars': nchars}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


if __name__ == '__main__':
    with Path(PARAMS).open() as f:
        params = json.load(f)

    params['words'] = str(Path(DATADIR, 'vocab.words.txt'))
    params['chars'] = str(Path(DATADIR, 'vocab.chars.txt'))
    params['tags'] = str(Path(DATADIR, 'vocab.tags.txt'))
    params['glove'] = str(Path(DATADIR, 'glove.npz'))

    estimator = tf.estimator.Estimator(model_fn, MODELDIR, params=params)
    export_path = estimator.export_saved_model('saved_model', serving_input_receiver_fn)
