- 컴퓨터는 텍스트보다는 숫자를 더 잘 처리가능
- 첫 단계로 각 단어를 고유한 정수에 맵핑(mapping)시키는 전처리 작업이 필요
# 1. 정수 인코딩(Integer Encoding)
## 1) dictionary 사용하기
```
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_text = "A barber is a person. a barber is good person.
    a barber is huge person. he Knew A Secret!
    The Secret He Kept is huge secret. Huge secret. His barber kept his word.
    a barber kept his word. His barber kept his secret.
    But keeping and keeping such a huge secret to himself was driving the barber crazy.
    the barber went up a huge mountain."
# 문장 토큰화
sentences = sent_tokenize(raw_text)
print(sentences)

[output]
['A barber is a person.', 'a barber is good person.', 'a barber is huge person.',
  'he Knew A Secret!', 'The Secret He Kept is huge secret.',
  'Huge secret.', 'His barber kept his word.', 'a barber kept his word.',
  'His barber kept his secret.', 'But keeping and keeping such a huge secret to himself was driving the barber crazy.',
  'the barber went up a huge mountain.']
```

- 소문자화, 불용어 제거 등의 단어가 텍스트일 때만 할 수 있는 최대한의 전처리를 끝내놔야 한다
```
vocab = {}
preprocessed_sentences = []
stop_words = set(stopwords.words('english'))

for sentence in sentences:
    # 단어 토큰화
    tokenized_sentence = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence: 
        word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄인다.
        if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거한다.
            if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거한다.
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0 
                vocab[word] += 1
    preprocessed_sentences.append(result) 
print(preprocessed_sentences)

[output]
[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'],
 ['barber', 'kept', 'word'],['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
```
```
print('단어 집합 :',vocab)

[output]
단어 집합 : {'barber': 8, 'person': 3, 'good': 1, 'huge': 5, 'knew': 1, 'secret': 6, 'kept': 4, 'word': 2, 'keeping': 2, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1}
```
- 빈도수 리턴
```
# 'barber'라는 단어의 빈도수 출력
print(vocab["barber"])

[output]
8
```
- 빈도수가 높은 순서대로 정렬
```
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
print(vocab_sorted)

[output]
[('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3), ('word', 2), ('keeping', 2), ('good', 1), ('knew', 1), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)]
```
- 높은 빈도수를 가진 단어일수록 낮은 정수를 부여합니다. 정수는 1부터 부여
```
word_to_index = {}
i = 0
for (word, frequency) in vocab_sorted :
    if frequency > 1 : # 빈도수가 작은 단어는 제외.
        i = i + 1
        word_to_index[word] = i

print(word_to_index)

[output]
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7}
```
- 1의 인덱스를 가진 단어가 가장 빈도수가 높은 단어
- 전처리인 빈도수가 적은 단어를 제외시키는 작업을 수행
```
vocab_size = 5

# 인덱스가 5 초과인 단어 제거
words_frequency = [word for word, index in word_to_index.items() if index >= vocab_size + 1]

# 해당 단어에 대한 인덱스 정보를 삭제
for w in words_frequency:
    del word_to_index[w]
print(word_to_index)

[output]
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```
- 단어 집합에 존재하지 않는 단어들이 생기는 상황을 Out-Of-Vocabulary(단어 집합에 없는 단어) 문제
- word_to_index에 'OOV'란 단어를 새롭게 추가하고, 단어 집합에 없는 단어들은 'OOV'의 인덱스로 인코딩
```
word_to_index['OOV'] = len(word_to_index) + 1
print(word_to_index)

[output]
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'OOV': 6}
```
- word_to_index를 사용하여 sentences의 모든 단어들을 맵핑되는 정수로 인코딩
```
encoded_sentences = []
for sentence in preprocessed_sentences:
    encoded_sentence = []
    for word in sentence:
        try:
            # 단어 집합에 있는 단어라면 해당 단어의 정수를 리턴.
            encoded_sentence.append(word_to_index[word])
        except KeyError:
            # 만약 단어 집합에 없는 단어라면 'OOV'의 정수를 리턴.
            encoded_sentence.append(word_to_index['OOV'])
    encoded_sentences.append(encoded_sentence)
print(encoded_sentences)

[output]
[[1, 5], [1, 6, 5], [1, 3, 5], [6, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [6, 6, 3, 2, 6, 1, 6], [1, 6, 3, 6]]
```

## 2) Counter 사용하기
```
from collections import Counter
print(preprocessed_sentences)

[output]
[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'],
 ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
```
- 단어 집합(vocabulary)을 만들기 위해서 sentences에서 문장의 경계인 [, ]를 제거하고 단어들을 하나의 리스트로 생성
```
# words = np.hstack(preprocessed_sentences)으로도 수행 가능.
all_words_list = sum(preprocessed_sentences, [])
print(all_words_list)

[output]
['barber', 'person', 'barber', 'good', 'person', 'barber', 'huge', 'person', 'knew', 'secret', 'secret', 'kept', 'huge', 'secret', 'huge', 'secret', 'barber', 'kept', 'word', 'barber', 'kept', 'word', 'barber', 'kept', 'secret', 'keeping', 'keeping',
 'huge', 'secret', 'driving', 'barber', 'crazy', 'barber', 'went', 'huge', 'mountain']
```
- Counter()의 입력으로 사용하면 중복을 제거하고 단어의 빈도수를 기록
```
# 파이썬의 Counter 모듈을 이용하여 단어의 빈도수 카운트
vocab = Counter(all_words_list)
print(vocab)

[output]
Counter({'barber': 8, 'secret': 6, 'huge': 5, 'kept': 4, 'person': 3, 'word': 2, 'keeping': 2, 'good': 1, 'knew': 1, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1})
```
- 키(key)로, 단어에 대한 빈도수가 값(value)으로 저장되어져 있습니다. vocab에 단어를 입력하면 빈도수를 리턴
```
print(vocab["barber"]) # 'barber'라는 단어의 빈도수 출력

[output]
8
```
- most_common()는 상위 빈도수를 가진 주어진 수의 단어만을 리턴
```
vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
vocab

[output]
[('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]
```
- 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스를 부여
```
word_to_index = {}
i = 0
for (word, frequency) in vocab :
    i = i + 1
    word_to_index[word] = i

print(word_to_index)

[output]
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```

## 3) NLTK의 FreqDist 사용하기
- NLTK에서는 빈도수 계산 도구인 FreqDist()를 지원
- Counter()랑 같은 방법으로 사용
```
from nltk import FreqDist
import numpy as np

# np.hstack으로 문장 구분을 제거
vocab = FreqDist(np.hstack(preprocessed_sentences))
```

- 단어를 키(key)로, 단어에 대한 빈도수가 값(value)으로 저장
- vocab에 단어를 입력하면 빈도수를 리턴
```
print(vocab["barber"]) # 'barber'라는 단어의 빈도수 출력

[output]
8
```
```
vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
print(vocab)

[output]
[('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]
```
- 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스를 부여합니다. 그런데 이번에는 enumerate()를 사용하여 좀 더 짧은 코드로 인덱스를 부여
```
word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
print(word_to_index)

[output]
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```
## 4) enumerate 이해하기
- enumerate()는 순서가 있는 자료형(list, set, tuple, dictionary, string)을 입력으로 받아 인덱스를 순차적으로 함께 리턴
```
test_input = ['a', 'b', 'c', 'd', 'e']
for index, value in enumerate(test_input): # 입력의 순서대로 0부터 인덱스를 부여함.
  print("value : {}, index: {}".format(value, index))
 
[output]
value : a, index: 0
value : b, index: 1
value : c, index: 2
value : d, index: 3
value : e, index: 4
```
# 2. 케라스(Keras)의 텍스트 전처리
- 케라스(Keras)는 기본적인 전처리를 위한 도구들을 제공
```
from tensorflow.keras.preprocessing.text import Tokenizer

preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'],
 ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()

# fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성.
tokenizer.fit_on_texts(preprocessed_sentences) 
```
- fit_on_texts는 입력한 텍스트로부터 단어 빈도수가 높은 순으로 낮은 정수 인덱스를 부여
```
print(tokenizer.word_index)

[output]
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9,
 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}
```
- 각 단어가 카운트를 수행하였을 때 몇 개였는지를 보고자 한다면 word_counts를 사용
```
print(tokenizer.word_counts)

[output]
OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2),
 ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])
```
- texts_to_sequences()는 입력으로 들어온 코퍼스에 대해서 각 단어를 이미 정해진 인덱스로 변환
```
print(tokenizer.texts_to_sequences(preprocessed_sentences))

[output]
[[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2],
 [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
```
- 케라스 토크나이저에서는 tokenizer = Tokenizer(num_words=숫자)와 같은 방법으로 빈도수가 높은 상위 몇 개의 단어만 사용하겠다고 지정 가능
```
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1) # 상위 5개 단어만 사용
tokenizer.fit_on_texts(preprocessed_sentences)
```
- num_words에서 +1을 더해서 값을 넣어주는 이유는 num_words는 숫자를 0부터 카운트
- 1 ~ 5번 단어까지 사용하고 싶다면 num_words에 숫자 5를 넣어주는 것이 아니라 5+1인 값 입력
- 숫자 0에 지정된 단어가 존재하지 않는데도 케라스 토크나이저가 숫자 0까지 단어 집합의 크기로 산정하는 이유는 자연어 처리에서 패딩(padding)이라는 작업 때문
```
print(tokenizer.word_index)

[output]
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}

print(tokenizer.word_counts)

[output]
OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])
```

- 실제 적용은 texts_to_sequences를 사용할 때 적용
```
print(tokenizer.texts_to_sequences(preprocessed_sentences))

[output]
[[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]
```
- word_index와 word_counts에서도 지정된 num_words만큼의 단어만 남기고 싶다면 아래의 코드도 방법
```
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)

vocab_size = 5
words_frequency = [word for word, index in tokenizer.word_index.items() if index >= vocab_size + 1] 

# 인덱스가 5 초과인 단어 제거
for word in words_frequency:
    del tokenizer.word_index[word] # 해당 단어에 대한 인덱스 정보를 삭제
    del tokenizer.word_counts[word] # 해당 단어에 대한 카운트 정보를 삭제

print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(preprocessed_sentences))

[output]
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
OrderedDict([('barber', 8), ('person', 3), ('huge', 5), ('secret', 6), ('kept', 4)])
[[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2], [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]
```
- 케라스 토크나이저는 기본적으로 단어 집합에 없는 단어인 OOV에 대해서는 단어를 정수로 바꾸는 과정에서 아예 단어를 제거한다는 특징
-  단어 집합에 없는 단어들은 OOV로 간주하여 보존하고 싶다면 Tokenizer의 인자 oov_token을 사용
```
# 숫자 0과 OOV를 고려해서 단어 집합의 크기는 +2
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 2, oov_token = 'OOV')
tokenizer.fit_on_texts(preprocessed_sentences)
```
- oov_token을 사용하기로 했다면 케라스 토크나이저는 기본적으로 'OOV'의 인덱스를 1
```
print('단어 OOV의 인덱스 : {}'.format(tokenizer.word_index['OOV']))

[output]
단어 OOV의 인덱스 : 1
```
```
print(tokenizer.texts_to_sequences(preprocessed_sentences))

[output]
[[2, 6], [2, 1, 6], [2, 4, 6], [1, 3], [3, 5, 4, 3], [4, 3], [2, 5, 1], [2, 5, 1], [2, 5, 3], [1, 1, 4, 3, 1, 2, 1], [2, 1, 4, 1]]
```
- 상위 5개의 단어는 2 ~ 6까지의 인덱스를 가졌으며, 그 외 단어 집합에 없는 'good'과 같은 단어들은 전부 'OOV'의 인덱스인 1로 인코딩
