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
    def __encode(cls, text: str):
        text = cls.__SLASH_PATTERN.sub(cls.__SLASH + '/' + '\g<2>', text)
        text = cls.__PLUS_PATTERN.sub(cls.__PLUS + '/' + '\g<2>', text)
        return text

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

    def match(self, targets):
        if Morpheme.__matcher is None:
            Morpheme.__matcher = Morpheme.Matcher(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), './res/complex_morpheme.dict'))

        tags = [None for _ in range(len(targets))]
        index = 0
        for i in range(len(targets)):
            for j in range(index, len(self)):
                if targets[i].lower() == self.__tags[j][0].lower():
                    tags[i] = self.__tags[j][1]
                    index += 1
                    break
            if tags[i] is None:
                tag, matches = Morpheme.__matcher.match(targets[i], self.__tags[index:])
                if tag is None:
                    raise ValueError('Not found patterns for %s, %s' % (targets, self))
                else:
                    tags[i] = tag
                    index += len(matches)

        return tags

    class Matcher:

        def __init__(self, path):
            self.__dict = {}
            with open(path, encoding='utf-8') as fp:
                for line in fp:
                    items = line.strip().split(' ')
                    key = items[0]
                    value = [ch for ch in items[1]]

                    if key not in self.__dict:
                        self.__dict[key] = []
                    self.__dict[key].append(value)

            for value in self.__dict.values():
                value.sort(key=lambda s: len(s))
                value.reverse()

        def match(self, ch, tags):
            patterns = self.__dict.get(ch)
            if patterns:
                for pattern in patterns:
                    for i in range(len(tags)):
                        if tags[i][0] == pattern[0] and len(tags) - i >= len(pattern):
                            subset = tags[i:(i + len(pattern))]
                            if pattern == [ch for ch, tag in subset]:
                                return ''.join(tag for ch, tag in subset), subset
            elif ch == '으':
                return 'mg', []

            return None, None
