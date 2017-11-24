class Corpus:
    def __init__(self, path):
        self.__items = []
        with open(path, encoding='utf-8') as fp:
            for line in fp:
                line = line.strip().split('\t')
                if len(line[0]) != len(line[1].split(' ')):
                    raise ValueError('Lengths should be same. text:%d, tag:%d' % (len(line[0]), len(line[1])))

                self.__items.append({
                    'text': line[0],
                    'tag': line[1],
                    'length': len(line[0])
                })

    def __len__(self) -> int:
        return len(self.__items)

    def items(self):
        return self.__items
