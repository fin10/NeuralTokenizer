import random

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


class NeuralPosTagger:
    __params = {}

    def __init__(self, output_size, vocab_size, max_length, model_dir):
        self.__params.update({
            'output_size': output_size,
            'vocab_size': vocab_size,
            'max_length': max_length,
            'batch_size': 2000,
            'cell_size': 150,
            'char_embedding_size': 100,
            'learning_rate': 0.001,
        })

        self.__estimator = tf.estimator.Estimator(
            model_fn=self.__model_fn,
            model_dir=model_dir,
            config=tf.estimator.RunConfig(
                save_summary_steps=10,
                save_checkpoints_steps=10,
            ),
            params=self.__params,
        )

    @classmethod
    def __input_fn(cls, inputs, batch_size=-1, shuffle=False):
        features = {
            'ids': [],
            'lengths': [],
            'masks': []
        }
        labels = []

        if shuffle:
            random.shuffle(inputs)

        if 0 < batch_size < len(inputs):
            inputs = inputs[:batch_size]

        max_length = cls.__params['max_length']
        for i in inputs:
            features['ids'].append(i['x'])
            features['lengths'].append(i['length'])
            features['masks'].append([1.0 if n < i['length'] else 0.0 for n in range(max_length)])
            labels.append(i['y'])

        features['ids'] = tf.constant(np.asarray(features['ids']))
        features['lengths'] = tf.constant(features['lengths'])
        features['masks'] = tf.constant(features['masks'])
        labels = tf.constant(np.asarray(labels))

        return features, labels

    @staticmethod
    def __model_fn(features, labels, mode, params):
        cell_size = params['cell_size']
        output_size = params['output_size']
        vocab_size = params['vocab_size']
        embedding_size = params['char_embedding_size']
        learning_rate = params['learning_rate']
        keep_prob = 1.0 if mode != tf.contrib.learn.ModeKeys.TRAIN else 0.5

        ids = features['ids']
        lengths = features['lengths']
        masks = features['masks']

        char_embeddings = tf.get_variable(
            name='char_embeddings',
            shape=[vocab_size, embedding_size],
            initializer=tf.random_uniform_initializer(-1, 1)
        )

        inputs = tf.nn.embedding_lookup(char_embeddings, ids)

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
        if mode != tf.estimator.ModeKeys.PREDICT:
            onehot_labels = tf.one_hot(labels, output_size, dtype=tf.float32)
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=outputs,
                weights=masks
            )

            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(
                    labels=labels,
                    predictions=predictions,
                    weights=masks
                )
            }

        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=tf.train.get_global_step(),
                decay_steps=10,
                decay_rate=0.96
            )

            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                loss=loss,
                global_step=tf.train.get_global_step(),
            )

        return tf.estimator.EstimatorSpec(
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
        pivot = int(len(inputs) * 0.99)
        train_set = inputs[:pivot]
        dev_set = inputs[pivot:]

        class ValidationHook(tf.train.SessionRunHook):

            def __init__(self, estimator, input_fn, dataset):
                self.__every_n_steps = 100
                self.__estimator = estimator
                self.__input_fn = input_fn
                self.__dataset = dataset

            @staticmethod
            def chunks(l, n):
                for i in range(0, len(l), n):
                    yield l[i:i + n]

            def before_run(self, run_context):
                graph = run_context.session.graph
                return tf.train.SessionRunArgs(tf.train.get_global_step(graph))

            def after_run(self, run_context, run_values):
                if run_values.results % self.__every_n_steps == 0:
                    accuracy = []
                    loss = []
                    for chunk in self.chunks(self.__dataset, 2000):
                        result = self.__estimator.evaluate(
                            input_fn=lambda: self.__input_fn(chunk),
                            steps=1,
                        )
                        accuracy.append(result['accuracy'])
                        loss.append(result['loss'])
                    print('#%d Accuracy: %s, Loss: %s' % (run_values.results, np.mean(accuracy), np.mean(loss)))

        self.__estimator.train(
            input_fn=lambda: self.__input_fn(train_set, batch_size=self.__params['batch_size'], shuffle=True),
            hooks=[ValidationHook(self.__estimator, self.__input_fn, dev_set)],
        )

    def eval(self, inputs):
        return self.__estimator.evaluate(
            input_fn=lambda: self.__input_fn(inputs),
            steps=1,
        )['accuracy']

    def predict(self, inputs):
        return []
