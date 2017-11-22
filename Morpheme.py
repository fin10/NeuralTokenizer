import os
import re


class Morpheme:
    __MORPHEME_PATTERN = re.compile('([^/]+)/([a-z]+)')

    __SLASH_PATTERN = re.compile('(/)/([a-z]+)')
    __PLUS_PATTERN = re.compile('(\+)/([a-z]+)')

    __SLASH = '_slash_'
    __PLUS = '_plus_'

    __matcher = None

    @classmethod
    def __encode(cls, labels: str):
        labels = cls.__SLASH_PATTERN.sub(cls.__SLASH + '/' + '\g<2>', labels)
        labels = cls.__PLUS_PATTERN.sub(cls.__PLUS + '/' + '\g<2>', labels)
        return labels

    @classmethod
    def __decode(cls, text: str):
        text = text.replace(cls.__SLASH, '/')
        text = text.replace(cls.__PLUS, '+')
        return text

    def __init__(self, raw: str):
        self.__tags = []

        raw = self.__encode(raw)
        for chunk in raw.split('+'):
            m = self.__MORPHEME_PATTERN.match(chunk)
            text = m.group(1)
            tag = m.group(2)

            text = self.__decode(text)
            for ch in text:
                self.__tags.append((ch, tag))

    def __len__(self) -> int:
        return len(self.__tags)

    def __str__(self):
        return '+'.join(['%s/%s' % (text, tag) for text, tag in self.__tags])

    def tags(self):
        return self.__tags

    def match(self, chars):
        if Morpheme.__matcher is None:
            Morpheme.__matcher = Morpheme.Matcher(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), './res/complex_morpheme.dict'))

        tags = []
        index = 0
        for ch in chars:
            if ch == self.__tags[index][0]:
                tags.append(self.__tags[index][1])
                index += 1
            else:
                tag, morpheme = self.__matcher.match(ch, self.__tags[index:])
                if tag is None:
                    raise ValueError('%s, %s' % (chars, self))
                tags.append(tag)

                if morpheme:
                    index += len(morpheme)

        return tags

    class Matcher:

        def __init__(self, path):
            self.__dict = {}
            with open(path) as fp:
                for line in fp:
                    items = line.strip().split(' ')
                    key = Morpheme(items[0])
                    value = Morpheme(items[1])
                    excepted = True if len(items) > 2 else False

                    text = key.tags()[0][0]
                    tag = key.tags()[0][1]
                    if text not in self.__dict:
                        self.__dict[text] = Morpheme.ComplexMorpheme()
                    self.__dict[text].add(value, tag, excepted)

        def match(self, ch, tags):
            cm = self.__dict.get(ch)
            if cm:
                return cm.find(tags)
            else:
                return None, None

    class ComplexMorpheme:

        def __init__(self):
            self.__morphemes = []

        def add(self, morpheme, tag, excepted):
            self.__morphemes.append((morpheme, tag, excepted))

        def find(self, tags):
            for morpheme, tag, excepted in self.__morphemes:
                found = True
                for i in range(len(morpheme)):
                    try:
                        if morpheme.tags()[i] != tags[i]:
                            found = False
                    except IndexError:
                        found = False

                if found:
                    return tag, morpheme if not excepted else None

            return None, None
