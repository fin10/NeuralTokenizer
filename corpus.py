import os

from Morpheme import Morpheme
from complex_morpheme_dict import ComplexMorphemeDict


class Corpus:
    __complex_morpheme_dict = ComplexMorphemeDict(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), './dict/complex_morpheme.dict'))

    def __init__(self, texts: list, labels: list):
        self.__items = []

        for d in zip(texts, labels):
            text = []
            labels = []

            text_tokens = d[0].split(' ')
            morpheme_tokens = d[1].split(' ')
            for text_token, morpheme_token in zip(text_tokens, morpheme_tokens):
                morpheme = Morpheme(morpheme_token)

                for ch in text_token:
                    if ch == morpheme.text()[0]:
                        ch, tag = morpheme.pop(0)
                        text.append(ch)
                        labels.append(tag)
                    else:
                        tag, index = self.__complex_morpheme_dict.find(ch, morpheme)
                        if tag is None:
                            raise ValueError('Not found pattern. %s, %s' % (text_token, morpheme_token))
                        text.append(ch)
                        labels.append(tag)
                        for _ in range(index):
                            morpheme.pop(0)

                text.append(' ')
                labels.append('o')

            text = text[0:-1]
            labels = labels[0:-1]

            for i in range(len(labels)):
                if labels[i] == 'o':
                    continue

                if i == 0 or not labels[i - 1].endswith(labels[i]):
                    labels[i] = 'b-' + labels[i]
                else:
                    labels[i] = 'i-' + labels[i]

            self.__items.append(Corpus.Item(text, labels))

    def __len__(self) -> int:
        return len(self.__items)

    def get(self, position):
        return self.__items[position]

    class Item:
        def __init__(self, text: list, labels: list):
            if len(text) != len(labels):
                raise ValueError('Lengths should be same. text:%d, labels: %d' % (len(text), len(labels)))

            self.__text = text
            self.__labels = labels
            self.__length = len(text)

        def text(self):
            return self.__text

        def labels(self):
            return self.__labels

        def length(self):
            return self.__length


if __name__ == '__main__':
    with open('./data/sejong.txt') as fp:
        texts = [line.strip() for line in fp]

    with open('./data/sejong.pos') as fp:
        labels = [line.strip() for line in fp]

    print('texts:%d, labels:%d' % (len(texts), len(labels)))
    corpus = Corpus(texts, labels)
    print('corpus: %d' % len(corpus))
