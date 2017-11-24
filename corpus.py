from morpheme import Morpheme


class Corpus:

    def __init__(self, texts: list, labels: list):
        self.__items = []
        errors = set()
        for text, label in zip(texts, labels):
            chars = []
            tags = []

            try:
                text_tokens = text.split(' ')
                morpheme_tokens = label.split(' ')
                for text_token, morpheme_token in zip(text_tokens, morpheme_tokens):
                    chars += [ch for ch in text_token]
                    tags += Morpheme(morpheme_token).match(text_token)
                    chars.append(' ')
                    tags.append('o')

                chars = chars[0:-1]
                tags = tags[0:-1]

                for i in range(len(tags)):
                    if tags[i] == 'o':
                        continue

                    if i == 0 or not tags[i - 1].endswith(tags[i]):
                        tags[i] = 'b-' + tags[i]
                    else:
                        tags[i] = 'i-' + tags[i]

                self.__items.append(Corpus.Item(chars, tags))

            except ValueError as e:
                errors.add(e)

        print('Pattern Errors: %d' % len(errors))
        with open('./pattern_errors.txt', mode='w', encoding='utf-8') as fp:
            fp.write('\n'.join([str(error) for error in errors]))

    def __len__(self) -> int:
        return len(self.__items)

    def export(self):
        with open('./corpus.output', mode='w', encoding='utf-8') as fp:
            for item in self.__items:
                fp.write('%s\t%s\n' % (''.join(item.text()), ' '.join(item.labels())))

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

    corpus.export()
