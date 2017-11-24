import random

import numpy as np
import tensorflow as tf


class NeuralPosTagger:
    __params = {}

    def __init__(self, output_size, vocab_size, max_length, model_dir):
        self.__params.update({
            'output_size': output_size,
            'vocab_size': vocab_size,
            'max_length': max_length,
            'batch_size': 100,
            'cell_size': 20,
            'char_embedding_size': 100,
            'learning_rate': 0.001,
        })

        self.__estimator = tf.contrib.learn.Estimator(
            model_fn=self.__model_fn,
            params=self.__params,
            config=tf.contrib.learn.RunConfig(
                save_checkpoints_steps=10,
                save_summary_steps=10,
            ),
            model_dir=model_dir
        )

    @classmethod
    def __input_fn(cls, inputs, batch_size=-1, shuffle=False):
        features = {
            'x': [],
            'lengths': [],
            'masks': []
        }

        targets = []

        if shuffle:
            random.shuffle(inputs)

        if 0 < batch_size < len(inputs):
            inputs = inputs[:batch_size]

        max_length = cls.__params['max_length']
        for i in inputs:
            features['x'].append(i['x'])
            features['lengths'].append(i['length'])
            features['masks'].append([1.0 if n < i['length'] else 0.0 for n in range(max_length)])
            targets.append(i['y'])

        features['x'] = tf.constant(np.asarray(features['x']))
        features['lengths'] = tf.constant(features['lengths'])
        features['masks'] = tf.constant(features['masks'])
        targets = tf.constant(np.asarray(targets))

        return features, targets

    @staticmethod
    def __model_fn(features, target, mode, params):
        cell_size = params['cell_size']
        output_size = params['output_size']

        vocab_size = params['vocab_size']
        embedding_size = params['char_embedding_size']

        learning_rate = params['learning_rate']
        keep_prob = 1.0 if mode != tf.contrib.learn.ModeKeys.TRAIN else 0.5

        x = features['x']
        lengths = features['lengths']
        masks = features['masks']

        char_embeddings = tf.get_variable(
            name='char_embeddings',
            shape=[vocab_size, embedding_size],
            initializer=tf.random_uniform_initializer(-1, 1)
        )

        inputs = tf.nn.embedding_lookup(char_embeddings, x)

        def rnn_cell(cell_size):
            cell = tf.contrib.rnn.GRUCell(cell_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            return cell

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell(cell_size),
            cell_bw=rnn_cell(cell_size),
            inputs=inputs,
            sequence_length=lengths,
            dtype=tf.float32
        )

        outputs = outputs[0] + outputs[1]

        outputs = tf.contrib.layers.fully_connected(
            inputs=outputs,
            num_outputs=output_size,
        )

        predictions = tf.argmax(outputs, 2)

        loss = None
        eval_metric_ops = None
        if mode != tf.contrib.learn.ModeKeys.INFER:
            onehot_labels = tf.one_hot(target, output_size, dtype=tf.float32)
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=outputs,
                weights=masks
            )

            eval_metric_ops = {
                'accuracy': tf.contrib.metrics.streaming_accuracy(
                    labels=target,
                    predictions=predictions,
                    weights=masks
                )
            }

        train_op = None
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=tf.train.get_global_step(),
                decay_steps=10,
                decay_rate=0.96
            )

            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=learning_rate,
                optimizer='Adam'
            )

        return tf.contrib.learn.ModelFnOps(
            mode=mode,
            predictions={
                'predictions': predictions
            },
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )

    def fit(self, inputs):
        random.shuffle(inputs)
        pivot = int(len(inputs) * 0.8)
        train_set = inputs[:pivot]
        dev_set = inputs[pivot:]

        validation_metrics = {
            'accuracy': tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key='predictions',
            )
        }

        monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=lambda: self.__input_fn(dev_set),
            eval_steps=1,
            every_n_steps=10,
            metrics=validation_metrics,
            early_stopping_metric='loss',
            early_stopping_metric_minimize=True,
            early_stopping_rounds=100
        )

        self.__estimator.fit(
            input_fn=lambda: self.__input_fn(train_set, batch_size=self.__params['batch_size'], shuffle=True),
            monitors=[monitor],
        )

    def eval(self, inputs):
        return self.__estimator.evaluate(
            input_fn=lambda: self.__input_fn(inputs),
            steps=1,
        )['accuracy']

    def predict(self, inputs):
        return []
