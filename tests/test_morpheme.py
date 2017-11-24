import unittest

from morpheme import Morpheme


class MorphemeTest(unittest.TestCase):

    def test(self):
        test_case = '421/nb+년/nu+째/xn'
        morpheme = Morpheme(test_case)
        self.assertEqual([ch for ch in '421년째'], [ch for ch, tag in morpheme.tags()])
        self.assertEqual(['nb', 'nb', 'nb', 'nu', 'xn'], [tag for ch, tag in morpheme.tags()])

    def test_slash(self):
        test_case = 'TCP/ne+//sc+IP/ne+를/po'
        morpheme = Morpheme(test_case)
        self.assertEqual([ch for ch in 'TCP/IP를'], [ch for ch, tag in morpheme.tags()])
        self.assertEqual(['ne', 'ne', 'ne', 'sc', 'ne', 'ne', 'po'], [tag for ch, tag in morpheme.tags()])

    def test_plus(self):
        test_case = '(/sl++/so+)/sr+반파/nc'
        morpheme = Morpheme(test_case)
        self.assertEqual([ch for ch in '(+)반파'], [ch for ch, tag in morpheme.tags()])
        self.assertEqual(['sl', 'so', 'sr', 'nc', 'nc'], [tag for ch, tag in morpheme.tags()])

    def test_match(self):
        test_case = ('썼거니와', '쓰/vb+었/ep+거니와/ec')
        morpheme = Morpheme(test_case[1])
        result = morpheme.match(test_case[0])
        self.assertEqual(['vbep', 'ec', 'ec', 'ec'], result)

    def test_match2(self):
        test_case = ('끝내!', '끝내/vb+아/ef+!/sf')
        morpheme = Morpheme(test_case[1])
        result = morpheme.match(test_case[0])
        self.assertEqual(['vb', 'vb', 'sf'], result)

    def test_match3(self):
        test_case = ('선각자였습니다.', '선/nc+각/nc+자/nc+이/pp+었/ep+습/ef+니/ef+다/ef+./sf')
        morpheme = Morpheme(test_case[1])
        result = morpheme.match(test_case[0])
        self.assertEqual(['nc', 'nc', 'nc', 'ppep', 'ef', 'ef', 'ef', 'sf'], result)

    def test_match4(self):
        test_case = ('활용한다든가', '활/na+용/na+하/xv+ㄴ/ec+다/ec+든/ec+가/ec')
        morpheme = Morpheme(test_case[1])
        result = morpheme.match(test_case[0])
        self.assertEqual(['na', 'na', 'xvec', 'ec', 'ec', 'ec'], result)

    def test_match5(self):
        test_case = ('보십시오.', '보/vx+시/ep+ㅂ/ef+시/ef+오/ef+./sf')
        morpheme = Morpheme(test_case[1])
        result = morpheme.match(test_case[0])
        self.assertEqual(['vx', 'epef', 'ef', 'ef', 'sf'], result)
