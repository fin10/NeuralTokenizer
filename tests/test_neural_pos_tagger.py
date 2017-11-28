import unittest

from neural_pos_tagger import NeuralPosTagger
from paths import Paths


class NeuralPosTaggerTest(unittest.TestCase):
    __tagger = NeuralPosTagger(Paths.MODEL)

    def test_train(self):
        self.__tagger.train('../sejong_corpus.train')

    def test_evaluate(self):
        self.__tagger.evaluate('../sejong_corpus.test')

    def test_predict(self):
        test_case = list('그녀는 내 얼굴을 똑바로 보지 않았다.')
        result = self.__tagger.predict(test_case)
        print('Text : %s' % ''.join(test_case))
        print('Token: %s' % (','.join('[{}/{}]'.format(ch, tag) for ch, tag in zip(test_case, result))))
