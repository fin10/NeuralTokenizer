import unittest

from morpheme import Morpheme


class MorphemeTest(unittest.TestCase):
    def test(self):
        test_case = '421/nb+년/nu+째/xn'
        morpheme = Morpheme(test_case)
        self.assertEqual([ch for ch in '421년째'], morpheme.text())
        self.assertEqual(['nb', 'nb', 'nb', 'nu', 'xn'], morpheme.tag())

    def test_slash(self):
        test_case = 'TCP/ne+//sc+IP/ne+를/po'
        morpheme = Morpheme(test_case)
        self.assertEqual([ch for ch in 'TCP/IP를'], morpheme.text())
        self.assertEqual(['ne', 'ne', 'ne', 'sc', 'ne', 'ne', 'po'], morpheme.tag())

    def test_plus(self):
        test_case = '(/sl++/so+)/sr+반파/nc'
        morpheme = Morpheme(test_case)
        self.assertEqual([ch for ch in '(+)반파'], morpheme.text())
        self.assertEqual(['sl', 'so', 'sr', 'nc', 'nc'], morpheme.tag())
