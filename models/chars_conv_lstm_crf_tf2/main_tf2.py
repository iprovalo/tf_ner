"""GloVe Embeddings + chars conv and max pooling + bi-LSTM + CRF - TF2

Original Author: Guillaume Genthial
Modified by: Ivan Provalov
Project: Tensorflow 2 Upgraded chars_conv_lstm_crf (derived from chars_conv_lstm_crf)
Original https://github.com/guillaumegenthial/tf_ner/blob/master/models/chars_conv_lstm_crf/
License: Apache License

"""


import functools
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tf_metrics_tf2 import precision, recall, f1

from masked_conv_tf2 import masked_conv1d_and_max

data_folder_name = './'

DATADIR = '../../data/example'
print(tf.__version__)


is_debug = False

# Logging
Path(f'{data_folder_name}/results').mkdir(exist_ok=True)
tf.compat.v1.logging.set_verbosity(logging.DEBUG)
tf.debugging.set_log_device_placement(True)  # Log device placement
handlers = [
    logging.FileHandler(f'{data_folder_name}/results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"

    # Chars
    chars = [[c.encode() for c in w] for w in line_words.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'\0'] * (max_len - l) for c, l in zip(chars, lengths)]
    return ((words, len(words)), (chars, lengths)), tags


def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words, tags, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    shapes = ((([None], ()),  # (words, nwords)
               ([None, None], [None])),  # (chars, nchars)
              [None])  # tags
    types = (((tf.string, tf.int32),
              (tf.string, tf.int32)),
             tf.string)
    defaults = ((('\0', 0),
                 ('\0', 0)),
                'O')
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, words, tags),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset


def model_fn(features, labels, mode, params):
    # For serving features are a bit different
    if isinstance(features, dict):
        features = ((features['words'], features['nwords']),
                    (features['chars'], features['nchars']))

    # Read vocabs and inputs

    dropout = params['dropout']
    (words, nwords), (chars, nchars) = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            filename=params['words'],
            key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            delimiter="\t"), params['num_oov_buckets'])

    vocab_chars = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            filename=params['chars'],
            key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            delimiter="\t"), params['num_oov_buckets'])

    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1
    with Path(params['chars']).open() as f:
        num_chars = sum(1 for _ in f) + params['num_oov_buckets']

    # Char Embeddings
    char_ids = vocab_chars.lookup(chars)

    initializer = tf.keras.initializers.GlorotUniform()
    variable = tf.Variable(initializer(shape=[num_chars + 1, params['dim_chars']], dtype=tf.float32),
                           name="chars_embeddings")

    char_embeddings = tf.nn.embedding_lookup(variable, char_ids)


    # Create a dropout layer
    dropout_layer = tf.keras.layers.Dropout(rate=dropout)
    # Apply the dropout layer
    char_embeddings = dropout_layer(char_embeddings, training=training)

    # char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout,
    #                                     training=training)

    # Char 1d convolution
    char_embeddings = masked_conv1d_and_max(char_embeddings, tf.sequence_mask(nchars), params['filters'], params['kernel_size'])

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']  # np.array
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

    # Concatenate Word and Char Embeddings
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)

    # Create a dropout layer
    dropout_layer = tf.keras.layers.Dropout(rate=dropout)
    # Apply the dropout layer
    embeddings = dropout_layer(embeddings, training=training)

    # LSTM
    if is_debug:
        embeddings = tf.compat.v1.Print(embeddings, [tf.shape(embeddings)], message="embeddings shape: ", summarize=30)
        embeddings = tf.compat.v1.Print(embeddings, [embeddings], message="embeddings: ", summarize=30)
    t = tf.transpose(embeddings, perm=[1, 0, 2])  # Need time-major

    # Define LSTM layers
    lstm_fw = tf.keras.layers.LSTM(params['lstm_size'], return_sequences=True, go_backwards=False, time_major=True)
    lstm_bw = tf.keras.layers.LSTM(params['lstm_size'], return_sequences=True, go_backwards=True, time_major=True)

    # Apply LSTM layers (forward and backward)
    if is_debug:
        t = tf.compat.v1.Print(t, [tf.shape(t)], message="t shape: ", summarize=30)
        t = tf.compat.v1.Print(t, [t], message="t: ", summarize=30)
        nwords = tf.compat.v1.Print(nwords, [tf.shape(nwords)], message="nwords shape: ", summarize=30)
        nwords = tf.compat.v1.Print(nwords, [nwords], message="nwords: ", summarize=30)
        nwords = tf.compat.v1.Print(nwords, [tf.shape(tf.sequence_mask(nwords))], message="nwords mask shape: ", summarize=30)
        nwords = tf.compat.v1.Print(nwords, [tf.sequence_mask(nwords)], message="nwords mask: ", summarize=30)

    lstm_mask = tf.sequence_mask(nwords)
    lstm_mask_transposed = tf.transpose(lstm_mask, perm=[1, 0])

    if is_debug:
        lstm_mask_transposed = tf.compat.v1.Print(lstm_mask_transposed, [tf.shape(lstm_mask_transposed)], message="lstm_mask_transposed shape: ", summarize=30)
        lstm_mask_transposed = tf.compat.v1.Print(lstm_mask_transposed, [lstm_mask_transposed], message="lstm_mask_transposed mask: ", summarize=30)

    output_fw = lstm_fw(t, mask=lstm_mask_transposed)
    output_bw = lstm_bw(t, mask=lstm_mask_transposed)

    # Concatenate outputs from both directions
    output = tf.concat([output_fw, output_bw], axis=-1)

    if is_debug:
        output = tf.compat.v1.Print(output, [tf.shape(output)], message="output shape: ", summarize=30)
        output = tf.compat.v1.Print(output, [output], message="output: ", summarize=30)

    output = tf.transpose(output, perm=[1, 0, 2])

    if is_debug:
        output = tf.compat.v1.Print(output, [tf.shape(output)], message="output shape: ", summarize=30)
        output = tf.compat.v1.Print(output, [output], message="output: ", summarize=30)

    # Create a dropout layer
    dropout_layer = tf.keras.layers.Dropout(rate=dropout)
    # Apply the dropout layer
    output = dropout_layer(output, training=training)

    # CRF
    # Create a Dense layer
    dense_layer = tf.keras.layers.Dense(num_tags)

    # Apply the Dense layer to the output tensor
    logits = dense_layer(output)

    initializer = tf.keras.initializers.GlorotUniform()
    crf_params = tf.Variable(initializer(shape=[num_tags, num_tags], dtype=tf.float32), name="crf")

    # Decode using CRF
    pred_ids, _ = tfa.text.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                filename=params['tags'],
                key_dtype=tf.int64,
                key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                value_dtype=tf.string,
                value_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                delimiter="\t"), default_value="UNK")
        pred_strings = reverse_vocab_tags.lookup(tf.cast(pred_ids, tf.int64))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
                filename=params['tags'],
                key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
                delimiter="\t"), num_oov_buckets=1)

        tags = vocab_tags.lookup(labels)

        # Compute the log-likelihood of the CRF
        log_likelihood, _ = tfa.text.crf_log_likelihood(logits, tags, nwords, crf_params)

        loss = tf.reduce_mean(-log_likelihood)

        # Metrics

        metrics = {
            'acc': tf.compat.v1.metrics.accuracy(tags, pred_ids),
            'precision': precision(tags, pred_ids, num_tags, indices, tf.sequence_mask(nwords)),
            'recall': recall(tags, pred_ids, num_tags, indices, tf.sequence_mask(nwords)),
            'f1': f1(tags, pred_ids, num_tags, indices, tf.sequence_mask(nwords)),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.compat.v1.train.AdamOptimizer()

            # Create the training operation
            train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_or_create_global_step())

            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    # Params
    params = {
        'dim_chars': 100,
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': NUMBER_OF_EPOCHS,
        'batch_size': BATCH_SIZE,
        'buffer': 15000,
        'filters': 50,
        'kernel_size': 3,
        'lstm_size': 100,
        'words': str(Path(DATADIR, 'vocab.words.txt')),
        'chars': str(Path(DATADIR, 'vocab.chars.txt')),
        'tags': str(Path(DATADIR, 'vocab.tags.txt')),
        'glove': str(Path(DATADIR, 'glove.npz'))
    }
    with Path(f'{data_folder_name}/results/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)


    def fwords(name):
        return str(Path(DATADIR, '{}.words.txt'.format(name)))


    def ftags(name):
        return str(Path(DATADIR, '{}.tags.txt'.format(name)))


    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'),
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('testa'), ftags('testa'))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, f'{data_folder_name}/results/model', cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


    # Write predictions to file
    def write_predictions(name):
        Path(f'{data_folder_name}/results/score').mkdir(parents=True, exist_ok=True)
        with Path('{}/results/score/{}.preds.txt'.format(data_folder_name, name)).open('wb') as f:
            test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
            golds_gen = generator_fn(fwords(name), ftags(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), (_, _)), tags = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')


    for name in ['train', 'testa', 'testb']:
        write_predictions(name)
