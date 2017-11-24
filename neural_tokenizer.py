import os
import shutil
from collections import Counter

import numpy as np
from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor

from corpus import Corpus
from neural_pos_tagger import NeuralPosTagger
from paths import Paths


class NeuralTokenizer:
    CHAR_PROCESSOR = 'char_processor.pkl'
    TAG_PROCESSOR = 'tag_processor.pkl'

    MAX_SENTENCE_LENGTH = 100

    def __init__(self):
        self.__char_processor = None
        self.__tag_processor = None
        self.__model = None

        self.__char_processor_path = os.path.join(Paths.MODEL, self.CHAR_PROCESSOR)
        self.__tag_processor_path = os.path.join(Paths.MODEL, self.TAG_PROCESSOR)

        if os.path.exists(Paths.MODEL):
            self.__char_processor = VocabularyProcessor.restore(self.__char_processor_path)
            self.__tag_processor = VocabularyProcessor.restore(self.__tag_processor_path)
            self.__model = self.__create_model()

    def __create_model(self):
        return NeuralPosTagger(
            output_size=len(self.__tag_processor.vocabulary_),
            vocab_size=len(self.__char_processor.vocabulary_),
            max_length=self.MAX_SENTENCE_LENGTH,
            model_dir=Paths.MODEL
        )

    @staticmethod
    def char_tokenizer_fn(raw):
        return [[ch for ch in raw]]

    @staticmethod
    def tag_tokenizer_fn(raw):
        return [raw.split(' ')]

    def train(self):
        training_corpus = Corpus('./sejong_corpus.train')
        test_corpus = Corpus('./sejong_corpus.test')

        if os.path.exists(Paths.MODEL):
            shutil.rmtree(Paths.MODEL)

        self.__char_processor = VocabularyProcessor(
            max_document_length=self.MAX_SENTENCE_LENGTH,
            tokenizer_fn=self.char_tokenizer_fn
        )

        self.__tag_processor = VocabularyProcessor(
            max_document_length=self.MAX_SENTENCE_LENGTH,
            tokenizer_fn=self.tag_tokenizer_fn
        )

        inputs = []
        for item in training_corpus.items():
            inputs.append({
                'x': list(self.__char_processor.transform(item['text']))[0],
                'y': list(self.__tag_processor.transform(item['tag']))[0],
                'length': item['length']
            })

        self.__char_processor.fit('')
        self.__tag_processor.fit('')

        os.makedirs(Paths.MODEL)
        self.__char_processor.save(self.__char_processor_path)
        self.__tag_processor.save(self.__tag_processor_path)

        print('Training: %d, Test: %d' % (len(training_corpus), len(test_corpus)))
        print('Char: %d, Tag: %d' % (len(self.__char_processor.vocabulary_),
                                     len(self.__tag_processor.vocabulary_)))

        lengths = [inp['length'] for inp in inputs]
        print('# Length')
        print('avg: %s, top: %s' % (np.mean(lengths), Counter(lengths).most_common(10)))
        print('longest: %d, shortest: %d' % (max(lengths), min(lengths)))

        self.__model = self.__create_model()
        self.__model.fit(inputs)
        print('Training completed.')

        inputs.clear()
        for item in test_corpus.items():
            inputs.append({
                'x': list(self.__char_processor.transform(item['text']))[0],
                'y': list(self.__tag_processor.transform(item['tag']))[0]
            })

        result = self.__model.eval(inputs)
        print('Accuracy: %.4f' % result)

    def tokenize(self, text):
        return []


if __name__ == '__main__':
    tokenizer = NeuralTokenizer()
    tokenizer.train()
