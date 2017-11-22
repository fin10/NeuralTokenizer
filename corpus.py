from morpheme import Morpheme


class Corpus:

    def __init__(self, texts: list, labels: list):
        self.__items = []
        errors = set()
        data_errors = []
        for d in zip(texts, labels):
            text = []
            labels = []

            text_tokens = d[0].split(' ')
            morpheme_tokens = d[1].split(' ')
            for text_token, morpheme_token in zip(text_tokens, morpheme_tokens):
                try:
                    labels += Morpheme(morpheme_token).match(text_token)
                    text += [ch for ch in text_token]
                except ValueError as e:
                    errors.add(e)
                except IndexError:
                    data_errors.append('%s, %s' % (text_token, morpheme_token))

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

        print('Data Errors: %d' % len(data_errors))
        with open('./data_errors.txt', mode='w', encoding='utf-8') as fp:
            fp.write('\n'.join(data_errors))

        print('Pattern Errors: %d' % len(errors))
        with open('./pattern_errors.txt', mode='w', encoding='utf-8') as fp:
            fp.write('\n'.join([str(error) for error in errors]))

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
