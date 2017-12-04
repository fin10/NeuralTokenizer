# NeuralTokenizer
TensorFlow로 구현한 한국어 토큰 분석기입니다.

## Dependencies
- Python 3
- TensorFlow 1.4

## Dataset
국립국어원에서 제공하는 [세종 코퍼스](https://ithub.korean.go.kr/user/total/referenceManager.do)를 사용 하였습니다.
[SejongCorpusParser](https://github.com/fin10/NeuralTokenizer/blob/master/sejong_corpus_parser.py)를 통해서 학습에 필요한 Dataset을 생성할 수 있습니다.

1. 국립국어원 언어정보나눔터에서 전산 처리용 세종 말뭉치 다운로드.
2. Project 폴더에서 data 폴더 생성 후 세종 말뭉치 파일 복사.
3. Python3 sejong_corpus_parser.py 실행.
4. data 폴더에 학습 및 테스트용 Dataset 파일 생성 확인.

## Training
NeuralPosTaggerTest의 test_train() 함수를 통해서 트레이닝을 시작할 수 있습니다.
트레이닝 된 모델은 model 폴더에 생성 됩니다.
