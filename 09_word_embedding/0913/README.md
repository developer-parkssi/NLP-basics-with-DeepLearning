- Word2Vec은 단어를 임베딩하는 워드 임베딩 알고리즘
- Doc2Vec은 Word2Vec을 변형하여 문서의 임베딩을 얻을 수 있도록 한 알고리즘
- 저자가 수집해놓은 전자공시시스템(Dart)에 올라와있는 각 회사의 사업보고서를 Doc2Vec을 통해서 학습
- 특정 회사와 사업 보고서가 유사한 회사들을 찾아본다

# 1. 공시 사업 보고서 로드 및 전처리

- 해당 실습은 형태소 분석기 Mecab의 원활한 설치를 위해서 구글의 Colab에서 진행했다고 가정
- 다른 형태소 분석기를 사용한다면 Colab에서 하지 않더라도 상관 X