from nltk.tokenize import sent_tokenize

text = ("His barber kept his word. But keeping such a huge secret to himself was driving him crazy."
        " Finally, the barber went up a mountain and almost to the edge of a cliff."
        " He dug a hole in the midst of some reeds. He looked about, to make sure no one was near.")
print('문장 토큰화1 :',sent_tokenize(text))

text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print('문장 토큰화2 :',sent_tokenize(text))

import kss

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화 :',kss.split_sentences(text))
