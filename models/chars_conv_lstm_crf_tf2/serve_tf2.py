"""Reload and serve a saved model

Original Author: Guillaume Genthial
Modified by: Ivan Provalov
Project: Tensorflow 2 Upgraded chars_conv_lstm_crf (derived from chars_conv_lstm_crf)
Original https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/
License: Apache License

"""

from pathlib import Path

import tensorflow as tf



LINE = 'hello world'


input_data = {
    'words': tf.constant([['hello', 'world']], dtype=tf.string),
    'nwords': tf.constant([2], dtype=tf.int32),
    'chars': tf.constant([[[c for c in 'hello'], [c for c in 'world']]], dtype=tf.string),
    'nchars': tf.constant([[5, 5]], dtype=tf.int32)
    }

def parse_fn(line):
    # Encode in Bytes for TF
    words = [w.encode() for w in line.strip().split()]
    words_tf = tf.constant([words], dtype=tf.string)
    nwords_tf = tf.constant([len(words)], dtype=tf.int32)

    # Chars
    chars = [[c.encode() for c in w] for w in line.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'\0'] * (max_len - l) for c, l in zip(chars, lengths)]
    chars_tf = tf.constant([chars], dtype=tf.string)
    nchars_tf = tf.constant([lengths], dtype=tf.int32)

    return {'words': words_tf, 'nwords': nwords_tf,
            'chars': chars_tf, 'nchars': nchars_tf}


if __name__ == '__main__':
    export_dir = 'saved_model'

    print(LINE)
    print(parse_fn(LINE))
    print(input_data)

    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    latest = tf.saved_model.load(latest)
    serving_fn = latest.signatures['serving_default']
    predictions = serving_fn(**parse_fn(LINE))
    print(predictions)
