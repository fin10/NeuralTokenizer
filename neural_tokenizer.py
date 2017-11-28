from neural_pos_tagger import NeuralPosTagger
from paths import Paths


class NeuralTokenizer:

    def __init__(self):
        self.__pos_tagger = NeuralPosTagger(Paths.MODEL)

    def train(self):
        self.__pos_tagger.train('./sejong_corpus.train')

    def tokenize(self, text):
        characters = list(text)
        result = self.__pos_tagger.predict(characters)

        tokens = []
        before = None
        for ch, tag in zip(characters, result):
            if tag.startswith('i-') and before and before[2:] == tag[2:]:
                tokens[-1] = (tokens[-1][0] + ch, tokens[-1][1])
            elif tag.startswith('b-'):
                tokens.append((ch, tag[2:]))
            else:
                tokens.append((ch, tag))
            before = tag

        return tokens
