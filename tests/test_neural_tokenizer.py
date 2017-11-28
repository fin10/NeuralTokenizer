import unittest

from neural_tokenizer import NeuralTokenizer


class NeuralTokenizerTest(unittest.TestCase):
    __tokenizer = NeuralTokenizer()

    def test_tokenize(self):
        test_case = '그녀는 내 얼굴을 똑바로 보지 않았다.'
        tokens = self.__tokenizer.tokenize(test_case)
        print('Text : %s' % test_case)
        print('Token: %s' % (' '.join('[{}/{}]'.format(chunk, tag) for chunk, tag in tokens)))
