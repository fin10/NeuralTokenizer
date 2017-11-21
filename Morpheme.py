import re


class Morpheme:
    __MORPHEME_PATTERN = re.compile('([^/]+)/([a-z]+)')

    __SLASH_PATTERN = re.compile('(/)/([a-z]+)')
    __PLUS_PATTERN = re.compile('(\+)/([a-z]+)')

    __SLASH = '_slash_'
    __PLUS = '_plus_'

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
        self.__text = []
        self.__tag = []

        raw = self.__encode(raw)
        for chunk in raw.split('+'):
            m = self.__MORPHEME_PATTERN.match(chunk)
            text = m.group(1)
            tag = m.group(2)

            text = self.__decode(text)
            self.__text += [ch for ch in text]
            self.__tag += [tag for _ in range(len(text))]

    def __len__(self) -> int:
        return len(self.__text)

    def text(self):
        return self.__text

    def tag(self):
        return self.__tag

    def pop(self, index=None):
        return self.__text.pop(index), self.__tag.pop(index)

    def get(self, index):
        return self.__text[index], self.__tag[index]
