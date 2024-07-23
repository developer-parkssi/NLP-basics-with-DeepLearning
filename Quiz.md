# 1. 다음 코드에 관한 설명을 맞추세요
```
vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size + 1) # 상위 5개 단어만 사용
tokenizer.fit_on_texts(preprocessed_sentences)
```
- 1 ~ 5번 단어까지 사용하고 싶다면 num_words에 숫자 5를 넣어주는 것이 아니라 5+1인 값 입력한다
- 숫자 0에 지정된 단어가 존재하지 않는데도 케라스 토크나이저가 숫자 0까지 단어 집합의 크기로 산정하는 이유는?


# 2. 원-핫 인코딩의 단점은 무엇인가요?
- 원-핫 인코딩의 단점을 두 가지씩 설명하세요.
