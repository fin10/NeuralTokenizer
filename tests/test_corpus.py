import unittest

from corpus import Corpus


class CorpusTest(unittest.TestCase):
    def test_constructor(self):
        test_case = ('노래비 세웠다.', '노래비/nc 세우/vb+었/ep+다/ef+./sf')
        corpus = Corpus([test_case])
        item = corpus.get(0)
        self.assertEqual(len(test_case[0]), item.length())
        self.assertEqual(['노', '래', '비', ' ', '세', '웠', '다', '.'], item.text())
        self.assertEqual(['b-nc', 'i-nc', 'i-nc', 'o', 'b-vb', 'b-vbep', 'b-ef', 'b-sf'], item.labels())

    def test_constructor2(self):
        test_case = ('처음 세워져', '처음/nc 세우/vb+어/ex+지/vx+어/ec')
        corpus = Corpus([test_case])
        item = corpus.get(0)
        self.assertEqual(len(test_case[0]), item.length())
        self.assertEqual(['처', '음', ' ', '세', '워', '져'], item.text())
        self.assertEqual(['b-nc', 'i-nc', 'o', 'b-vb', 'b-vbex', 'b-vxec'], item.labels())
