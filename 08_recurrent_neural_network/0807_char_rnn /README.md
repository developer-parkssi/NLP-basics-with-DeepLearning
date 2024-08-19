- 지금까지 배운 RNN은 전부 입력과 출력의 단위가 단어 벡터
- 입출력의 단위를 단어 레벨(word-level)에서 문자 레벨(character-level)로 변경하여 RNN을 구현 가능

![img.png](img.png)

- 문자 단위 RNN을 다 대 다(Many-to-Many) 구조로 구현한 경우, 다 대 일(Many-to-One) 구조로 구현한 경우 두 가지