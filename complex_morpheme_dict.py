from Morpheme import Morpheme


class ComplexMorphemeDict:
    def __init__(self, path):
        self.__dict = {}
        with open(path) as fp:
            for line in fp:
                items = line.strip().split(' ')
                key = Morpheme(items[0])
                value = Morpheme(items[1])

                text = ''.join(key.text())
                if text not in self.__dict:
                    self.__dict[text] = []
                self.__dict[text].append((value, key.tag()[0]))

    def find(self, ch, morpheme):
        values = self.__dict.get(ch)
        if values:
            for value in values:
                found = True
                for i in range(len(value[0])):
                    try:
                        if value[0].get(i) != morpheme.get(i):
                            found = False
                    except IndexError:
                        found = False

                if found:
                    return value[1], len(value[0])

        return None, -1
