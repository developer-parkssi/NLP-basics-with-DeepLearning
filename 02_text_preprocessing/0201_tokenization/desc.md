- 크롤링 등으로 얻어낸 코퍼스 데이터가 필요에 맞게 전처리되지 않은 상태라면, 해당 데이터를 사용하고자 하는 용도에 맞게 토큰화(tokenization) & 정제(cleaning) & 정규화(normalization)
# 1. 단어 토큰화(Word Tokenization)
- 구두점(punctuation)과 같은 문자는 제외시키는 간단한 단어 토큰화 작업
- 구두점이란 마침표(.), 컴마(,), 물음표(?), 세미콜론(;), 느낌표(!) 등과 같은 기호
```
[입력]
Time is an illusion. Lunchtime double so!

[출력]
"Time", "is", "an", "illustion", "Lunchtime", "double", "so"
```
# 2. 토큰화 중 생기는 선택의 순간
- 토큰화를 하다보면, 예상하지 못한 경우가 있어서 토큰화의 기준을 생각해봐야 하는 경우가 발생
- 영어권 언어에서 아포스트로피를(')가 들어가있는 단어들이 그 예시

# 3. 토큰화에서 고려해야할 사항
## 1) 구두점이나 특수 문자를 단순 제외해서는 안 된다.
- 구두점조차도 하나의 토큰으로 분류하기도 함
- 단어 자체에 구두점을 갖고 있는 경우   
ex) m.p.h나 Ph.D나 AT&T
- 숫자 사이에 컴마(,)가 들어가는 경우
## 2) 줄임말과 단어 내에 띄어쓰기가 있는 경우.
- 종종 영어권 언어의 아포스트로피(')는 압축된 단어를 다시 펼치는 역할   
ex) we're는 we are, rock 'n' roll
## 3) 표준 토큰화 예제
- 표준으로 쓰이고 있는 토큰화 방법 중 하나인 Penn Treebank Tokenization
1. 하이푼으로 구성된 단어는 하나로 유지한다.
2. doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리해준다.
```
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print('트리뱅크 워드토크나이저 :',tokenizer.tokenize(text))

[출력]
트리뱅크 워드토크나이저 : ['Starting', 'a', 'home-based', 'restaurant', 'may',
 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food',
  'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']
```
# 4. 문장 토큰화(Sentence Tokenization)
- 토큰의 단위가 문장(sentence)일 경우를 논의
- 코퍼스 내에서 문장 단위로 구분하는 작업으로 때로는 문장 분류(sentence segmentation)
- 마침표는 문장의 끝이 아니더라도 등장 가능   
ex) EX1) IP 192.168.56.31 서버에 들어가서 로그 파일 저장해서 aaa@gmail.com로 결과 좀 보내줘. 그 후 점심 먹으러 가자.
- NLTK에서는 영어 문장의 토큰화를 수행하는 sent_tokenize를 지원
```
from nltk.tokenize import sent_tokenize

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print('문장 토큰화1 :',sent_tokenize(text))

[출력]
문장 토큰화1 : ['His barber kept his word.',
 'But keeping such a huge secret to himself was driving him crazy.',
  'Finally, the barber went up a mountain and almost to the edge of a cliff.',
   'He dug a hole in the midst of some reeds.',
    'He looked about, to make sure no one was near.']
```
- 문장 중간에 마침표가 다수 등장하는 경우
```
text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print('문장 토큰화2 :',sent_tokenize(text))

[출력]
문장 토큰화2 : ['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
```
- 한국어에 대한 문장 토큰화 도구 또한 존재
```
import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :',kss.split_sentences(text))

[출력]
한국어 문장 토큰화 : ['딥 러닝 자연어 처리가 재미있기는 합니다.',
 '그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다.', '이제 해보면 알걸요?']
```
# 5. 한국어에서의 토큰화의 어려움.
- 영어는 띄어쓰기 토큰화를 수행해도 단어 토큰화가 잘 작동
- 한국어는 영어와는 달리 띄어쓰기만으로는 토큰화를 하기에 부족
- 한국어가 영어와는 다른 형태를 가지는 언어인 교착어라는 점에서 기인. 교착어란 조사, 어미 등을 붙여서 말을 만드는 언어
## 1) 교착어의 특성
- 한국어에는 조사라는 것이 존재
- '그가', '그에게', '그를', '그와', '그는'과 같이 다양한 조사가 '그'라는 글자 뒤에 띄어쓰기 없이 바로 붙게됨
- 한국어는 어절이 독립적인 단어로 구성되는 것이 아니라 조사 등의 무언가가 붙어있는 경우가 많아서 이를 전부 분리해줘야한다는 의미
- 한국어 토큰화에서는 형태소(morpheme) 란 개념을 반드시 이해. 형태소(morpheme)란 뜻을 가진 가장 작은 말의 단위
- 형태소에는 2가지가 존재
1. 자립 형태소: 접사, 어미, 조사와 상관없이 자립하여 사용할 수 있는 형태소. 그 자체로 단어가 된다. 체언(명사, 대명사, 수사), 수식언(관형사, 부사), 감탄사 등이 있다.
2. 의존 형태소: 다른 형태소와 결합하여 사용되는 형태소. 접사, 어미, 조사, 어간을 말한다.
```
문장 : 에디가 책을 읽었다

토큰화: ['에디가', '책을', '읽었다']

형태소 단위 분해
자립 형태소 : 에디, 책
의존 형태소 : -가, -을, 읽-, -었, -다
```
## 2) 한국어는 띄어쓰기가 영어보다 잘 지켜지지 않는다.
- 한국어의 경우 띄어쓰기가 지켜지지 않아도 글을 쉽게 이해할 수 있는 언어
- 결론적으로 한국어는 수많은 코퍼스에서 띄어쓰기가 무시되는 경우가 많아 자연어 처리가 어려워졌다는 것
# 6. 품사 태깅(Part-of-speech tagging)
- 단어의 의미를 제대로 파악하기 위해서는 해당 단어가 어떤 품사로 쓰였는지 보는 것이 주요 지표
```
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(text)

print('단어 토큰화 :',tokenized_sentence)
print('품사 태깅 :',pos_tag(tokenized_sentence))

[출력]
단어 토큰화 : ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
품사 태깅 : [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'),
 ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'),
  ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]
```
- Penn Treebank POG Tags에서 PRP는 인칭 대명사, VBP는 동사, RB는 부사, VBG는 현재부사, IN은 전치사, NNP는 고유 명사, NNS는 복수형 명사, CC는 접속사, DT는 관사를 의미
- OKT
```
from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 명사 추출 :',okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 

[출력]
OKT 형태소 분석 : ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
OKT 품사 태깅 : [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
OKT 명사 추출 : ['코딩', '당신', '연휴', '여행']
```
- 꼬꼬마
```
print('꼬꼬마 형태소 분석 :',kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('꼬꼬마 품사 태깅 :',kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('꼬꼬마 명사 추출 :',kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  

[출력]
꼬꼬마 형태소 분석 : ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
꼬꼬마 품사 태깅 : [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
꼬꼬마 명사 추출 : ['코딩', '당신', '연휴', '여행']
```