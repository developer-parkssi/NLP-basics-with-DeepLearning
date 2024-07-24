- 컴퓨터 또는 기계는 문자보다는 숫자를 더 잘 처리가능
- 이를 위해 자연어 처리에서는 문자를 숫자로 바꾸는 여러가지 기법 중 하나인 원핫 인코딩이 존재
- 단어 집합은 서로 다른 단어들의 집합
- 원-핫 인코딩을 위해서 먼저 해야할 일은 단어 집합을 만드는 일
- 텍스트의 모든 단어를 중복을 허용하지 않고 모아놓으면 이를 단어 집합
- 텍스트에 단어가 총 5,000개가 존재한다면, 단어 집합의 크기는 5,000
- 5,000개의 단어가 있는 이 단어 집합의 단어들마다 1번부터 5,000번까지 인덱스를 부여한다고 가정
- 각 단어에 고유한 정수 인덱스를 부여하였다고 가정. 이 숫자로 바뀐 단어들을 벡터로 다루고 싶다면 어떻게 해야할까?
# 1. 원-핫 인코딩(One-Hot Encoding)이란?
- 원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식   
-> 이렇게 표현된 벡터를 원-핫 벡터(One-Hot vector)
- 원-핫 인코딩을 두 가지 과정으로 정리
1. 정수 인코딩을 수행합니다. 다시 말해 각 단어에 고유한 정수를 부여
2. 표현하고 싶은 단어의 고유한 정수를 인덱스로 간주하고 해당 위치에 1을 부여하고, 다른 단어의 인덱스의 위치에는 0을 부여


ex) 문장 : 나는 자연어 처리를 배운다
```
from konlpy.tag import Okt  

okt = Okt()  
tokens = okt.morphs("나는 자연어 처리를 배운다")  
print(tokens)

[output]
['나', '는', '자연어', '처리', '를', '배운다']
```
- 각 토큰에 대해서 고유한 정수를 부여
```
word_to_index = {word : index for index, word in enumerate(tokens)}
print('단어 집합 :',word_to_index)

[output]
단어 집합 : {'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}
```
- 토큰을 입력하면 해당 토큰에 대한 원-핫 벡터를 만들어내는 함수
```
def one_hot_encoding(word, word_to_index):
  one_hot_vector = [0]*(len(word_to_index))
  index = word_to_index[word]
  one_hot_vector[index] = 1
  return one_hot_vector
```
- '자연어'라는 단어의 원-핫 벡터
```
one_hot_encoding("자연어", word_to_index)

[output]
[0, 0, 1, 0, 0, 0]  
```
- 자연어'는 정수 2이므로 원-핫 벡터는 인덱스 2의 값이 1이며, 나머지 값은 0인 벡터
# 2. 케라스(Keras)를 이용한 원-핫 인코딩(One-Hot Encoding)
- 케라스는 원-핫 인코딩을 수행하는 유용한 도구 to_categorical()를 지원

ex) text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"
```
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
print('단어 집합 :',tokenizer.word_index)

[output]
단어 집합 : {'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}
```
- 단어 집합(vocabulary)에 있는 단어들로만 구성된 텍스트가 있다면, texts_to_sequences()를 통해서 이를 정수 시퀀스로 변환가능
- 생성된 단어 집합 내의 일부 단어들로만 구성된 서브 텍스트인 sub_text를 만들어 확인
```
sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded)

[output]
[2, 5, 1, 6, 3, 7]
```
- 케라스는 정수 인코딩 된 결과로부터 원-핫 인코딩을 수행하는 to_categorical()를 지원
```
one_hot = to_categorical(encoded)
print(one_hot)

[output]
[[0. 0. 1. 0. 0. 0. 0. 0.] # 인덱스 2의 원-핫 벡터
 [0. 0. 0. 0. 0. 1. 0. 0.] # 인덱스 5의 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0.] # 인덱스 1의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 1. 0.] # 인덱스 6의 원-핫 벡터
 [0. 0. 0. 1. 0. 0. 0. 0.] # 인덱스 3의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 1.]] # 인덱스 7의 원-핫 벡터
```
# 3. 원-핫 인코딩(One-Hot Encoding)의 한계
- 단어의 개수가 늘어날 수록, 벡터를 저장하기 위해 필요한 공간이 계속 늘어난다는 단점
- 단어의 유사도를 표현하지 못한다는 단점
- 이러한 단점을 해결하기 위해 단어의 잠재 의미를 반영하여 다차원 공간에 벡터화 하는 기법으로 크게 두 가지
1. 카운트 기반의 벡터화 방법인 LSA(잠재 의미 분석), HAL 등
2. 예측 기반으로 벡터화하는 NNLM, RNNLM, Word2Vec, FastText 등
- 카운트 기반과 예측 기반 두 가지 방법을 모두 사용하는 방법으로 GloVe라는 방법이 존재