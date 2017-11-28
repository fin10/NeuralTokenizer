import os
import random
import shutil
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor

from corpus import Corpus

tf.logging.set_verbosity(tf.logging.INFO)


class NeuralPosTagger:
    MAX_SENTENCE_LENGTH = 100

    __CHAR_PROCESSOR = 'char_processor.pkl'
    __TAG_PROCESSOR = 'tag_processor.pkl'

    def __init__(self, model_dir):
        self.__params = {
            'max_length': self.MAX_SENTENCE_LENGTH,
            'batch_size': 1000,
            'epoch_size': 2,
            'cell_size': 300,
            'char_embedding_size': 300,
            'learning_rate': 0.001,
        }

        self.__model_path = model_dir
        self.__char_processor_path = os.path.join(self.__model_path, self.__CHAR_PROCESSOR)
        self.__tag_processor_path = os.path.join(self.__model_path, self.__TAG_PROCESSOR)

        self.__char_processor = None
        self.__tag_processor = None
        self.__estimator = None

        if os.path.exists(self.__model_path):
            self.__char_processor = VocabularyProcessor.restore(self.__char_processor_path)
            self.__tag_processor = VocabularyProcessor.restore(self.__tag_processor_path)

            self.__params.update({
                'output_size': len(self.__tag_processor.vocabulary_),
                'vocab_size': len(self.__char_processor.vocabulary_),
            })

            self.__estimator = self.__create_estimator()

    def __create_estimator(self):
        return tf.estimator.Estimator(
            model_fn=self.__model_fn,
            model_dir=self.__model_path,
            config=tf.estimator.RunConfig(
                save_summary_steps=10,
                save_checkpoints_steps=10,
            ),
            params=self.__params,
        )

    @staticmethod
    def char_tokenizer_fn(raw):
        return [[ch for ch in raw]]

    @staticmethod
    def tag_tokenizer_fn(raw):
        return [raw.split(' ')]

    def __input_fn(self, inputs, epoch=1):
        batch_size = self.__params['batch_size'] if self.__params['batch_size'] > 0 else len(inputs)
        max_length = self.__params['max_length']

        def gen(records: list):
            for record in records:
                yield {
                          'ids': record['x'],
                          'length': record['length'] if record['length'] < max_length else max_length,
                          'mask': [1 if n < record['length'] else 0 for n in range(max_length)],
                      }, record['y']

        dataset = tf.data.Dataset.from_generator(
            lambda: gen(inputs),
            ({'ids': tf.int32, 'length': tf.int32, 'mask': tf.int32}, tf.int32),
            ({
                 'ids': tf.TensorShape([max_length]),
                 'length': tf.TensorShape([]),
                 'mask': tf.TensorShape([max_length])
             }, tf.TensorShape([max_length]))
        )

        dataset = dataset.shuffle(batch_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(epoch)

        iterator = dataset.make_one_shot_iterator()
        features, label = iterator.get_next()

        return features, label

    @staticmethod
    def __model_fn(features, labels, mode, params):
        cell_size = params['cell_size']
        output_size = params['output_size']
        vocab_size = params['vocab_size']
        embedding_size = params['char_embedding_size']
        learning_rate = params['learning_rate']
        keep_prob = 1.0 if mode != tf.contrib.learn.ModeKeys.TRAIN else 0.5

        ids = features['ids']
        length = features['length']
        mask = features['mask']

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
            sequence_length=length,
            dtype=tf.float32
        )

        outputs = outputs[0] + outputs[1]

        outputs = tf.layers.dense(
            inputs=outputs,
            units=output_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

        predictions = tf.argmax(outputs, 2)

        loss = None
        eval_metric_ops = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            onehot_labels = tf.one_hot(labels, output_size, dtype=tf.float32)
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels,
                logits=outputs,
                weights=mask
            )

            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(
                    labels=labels,
                    predictions=predictions,
                    weights=mask
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
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )

    def train(self, corpus_path):
        self.__char_processor = VocabularyProcessor(
            max_document_length=self.MAX_SENTENCE_LENGTH,
            tokenizer_fn=NeuralPosTagger.char_tokenizer_fn
        )

        self.__tag_processor = VocabularyProcessor(
            max_document_length=self.MAX_SENTENCE_LENGTH,
            tokenizer_fn=NeuralPosTagger.tag_tokenizer_fn
        )

        items = []
        training_corpus = Corpus(corpus_path)
        for item in training_corpus.items():
            items.append({
                'x': list(self.__char_processor.transform(item['text']))[0],
                'y': list(self.__tag_processor.transform(item['tag']))[0],
                'length': item['length']
            })

        self.__char_processor.fit('')
        self.__tag_processor.fit('')

        if os.path.exists(self.__model_path):
            shutil.rmtree(self.__model_path)

        os.makedirs(self.__model_path)
        self.__char_processor.save(self.__char_processor_path)
        self.__tag_processor.save(self.__tag_processor_path)

        self.__params.update({
            'output_size': len(self.__tag_processor.vocabulary_),
            'vocab_size': len(self.__char_processor.vocabulary_),
        })

        self.__estimator = self.__create_estimator()

        print('Training: %d' % len(training_corpus))
        print(
            'Character: %d, Tag: %d' % (len(self.__char_processor.vocabulary_), len(self.__tag_processor.vocabulary_)))

        lengths = [item['length'] for item in items]
        print('# Length')
        print('avg: %s, top: %s' % (np.mean(lengths), Counter(lengths).most_common(10)))
        print('longest: %d, shortest: %d' % (max(lengths), min(lengths)))

        random.shuffle(items)
        pivot = int(len(items) * 0.8)
        train_set = items[:pivot]
        dev_set = items[pivot:]

        class ValidationHook(tf.train.SessionRunHook):

            def __init__(self, estimator, input_fn, dataset):
                self.__every_n_steps = 100
                self.__estimator = estimator
                self.__input_fn = input_fn
                self.__dataset = dataset

            def before_run(self, run_context):
                graph = run_context.session.graph
                return tf.train.SessionRunArgs(tf.train.get_global_step(graph))

            def after_run(self, run_context, run_values):
                if run_values.results % self.__every_n_steps == 0:
                    result = self.__estimator.evaluate(
                        input_fn=lambda: self.__input_fn(self.__dataset),
                    )
                    print('#%d Accuracy: %s, Loss: %s' % (run_values.results, result['accuracy'], result['loss']))

        self.__estimator.train(
            input_fn=lambda: self.__input_fn(train_set, epoch=self.__params['epoch_size']),
            hooks=[ValidationHook(self.__estimator, self.__input_fn, dev_set)],
        )
        print('Training completed.')

    def evaluate(self, corpus_path):
        test_set = []
        test_corpus = Corpus(corpus_path)
        for item in test_corpus.items():
            test_set.append({
                'x': list(self.__char_processor.transform(item['text']))[0],
                'y': list(self.__tag_processor.transform(item['tag']))[0],
                'length': item['length']
            })

        result = self.__estimator.evaluate(
            input_fn=lambda: self.__input_fn(test_set),
        )

        print('Test: %d' % len(test_corpus))
        print('Accuracy: %s, Loss: %s' % (result['accuracy'], result['loss']))

    def predict(self, characters: list):
        data_set = [{
            'x': list(self.__char_processor.transform(characters))[0],
            'y': [0 for _ in range(self.__params['max_length'])],
            'length': len(characters)
        }]

        result = list(self.__estimator.predict(
            input_fn=lambda: self.__input_fn(data_set),
        ))[0][:len(characters)]

        result = list(self.__tag_processor.reverse([result]))[0]
        result = result.split(' ')

        return result
