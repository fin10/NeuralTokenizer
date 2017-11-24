import os


class Paths:
    ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    MODEL = os.path.join(ROOT, 'model')
    RES = os.path.join(ROOT, 'res')
