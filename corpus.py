import re


class Corpus:
    __LABEL_PATTERN = re.compile('([^/]+)/([a-z]+)')

    def __init__(self, data: list):
        self.__items = []

        for d in data:
            text = []
            labels = []
            text_tokens = d[0].split(' ')
            label_tokens = d[1].split(' ')
            for text_token, label_token in zip(text_tokens, label_tokens):
                label_txt = []
                label_label = []
                for token in label_token.split('+'):
                    m = Corpus.__LABEL_PATTERN.match(token)
                    label_txt += [ch for ch in m.group(1)]
                    label_label += [m.group(2) for _ in range(len(m.group(1)))]

                for i in range(len(text_token)):
                    text.append(text_token[i])
                    if text_token[i] == label_txt[0]:
                        label_txt.pop(0)
                        labels.append(label_label.pop(0))
                    else:
                        label = ''
                        while text_token[i + 1] != label_txt[0]:
                            if len(label_txt) == 0:
                                break
                            label_txt.pop(0)
                            label += label_label.pop(0)

                        labels.append(label)

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

    def size(self):
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
