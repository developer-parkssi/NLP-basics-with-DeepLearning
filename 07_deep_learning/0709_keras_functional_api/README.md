- Sequential API는 여러층을 공유하거나 다양한 종류의 입력과 출력을 사용하는 등의 복잡한 모델을 만드는 일에는 한계

# 1. Sequential API로 만든 모델

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))
```

- 위와 같은 방식은 직관적이고 편리하지만 단순히 층을 쌓는 것만으로는 구현할 수 없는 복잡한 신경망을 구현X

# 2. Functional API로 만든 모델

## 1) 전결합 피드 포워드 신경망(Fully-connected FFNN)

- functional API에서는 입력 데이터의 크기(shape)를 인자로 입력층을 정의

```python

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


inputs = Input(shape=(10,))
hidden1 = Dense(64, activation='relu')(inputs)  # <- 새로 추가
hidden2 = Dense(64, activation='relu')(hidden1) # <- 새로 추가
output = Dense(1, activation='sigmoid')(hidden2) # <- 새로 추가
model = Model(inputs=inputs, outputs=output) # <- 새로 추가
```

- Input() 함수에 입력의 크기를 정의
- 이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당
- Model() 함수에 입력과 출력을 정의

- sequential API를 사용할 때와 마찬가지로 model.compile, model.fit 등을 사용 가능

```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(data, labels)
```

- 변수명을 달리해서 FFNN을 생성. 이번에는 은닉층과 출력층의 변수를 전부 x로 통일

```
inputs = Input(shape=(10,))
x = Dense(8, activation="relu")(inputs)
x = Dense(4, activation="relu")(x)
x = Dense(1, activation="linear")(x)
model = Model(inputs, x)
```

## 2) 선형 회귀(Linear Regression)

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

X = [1, 2, 3, 4, 5, 6, 7, 8, 9] # 공부하는 시간
y = [11, 22, 33, 44, 53, 66, 77, 87, 95] # 각 공부하는 시간에 맵핑되는 성적

inputs = Input(shape=(1,))
output = Dense(1, activation='linear')(inputs)
linear_model = Model(inputs, output)

sgd = optimizers.SGD(lr=0.01)

linear_model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
linear_model.fit(X, y, epochs=300)
```

## 3) 로지스틱 회귀(Logistic Regression)

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(3,))
output = Dense(1, activation='sigmoid')(inputs)
logistic_model = Model(inputs, output)
```

## 4) 다중 입력을 받는 모델(model that accepts multiple inputs)
- 다중 입력과 다중 출력을 가지는 모델

```python
# 최종 완성된 다중 입력, 다중 출력 모델의 예
model = Model(inputs=[a1, a2], outputs=[b1, b2, b3])

from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

# 두 개의 입력층을 정의
inputA = Input(shape=(64,))
inputB = Input(shape=(128,))

# 첫번째 입력층으로부터 분기되어 진행되는 인공 신경망을 정의
x = Dense(16, activation="relu")(inputA)
x = Dense(8, activation="relu")(x)
x = Model(inputs=inputA, outputs=x)

# 두번째 입력층으로부터 분기되어 진행되는 인공 신경망을 정의
y = Dense(64, activation="relu")(inputB)
y = Dense(32, activation="relu")(y)
y = Dense(8, activation="relu")(y)
y = Model(inputs=inputB, outputs=y)

# 두개의 인공 신경망의 출력을 연결(concatenate)
result = concatenate([x.output, y.output])

z = Dense(2, activation="relu")(result)
z = Dense(1, activation="linear")(z)

model = Model(inputs=[x.input, y.input], outputs=z)
```

## 5) RNN(Recurrence Neural Network) 은닉층 사용하기
- RNN 은닉층을 가지는 모델을 설계
- 하나의 특성(feature)에 50개의 시점(time-step)을 입력으로 받는 모델을 설계

```python
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model

inputs = Input(shape=(50,1))
lstm_layer = LSTM(10)(inputs)
x = Dense(10, activation='relu')(lstm_layer)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=output)
```

## 6) 다르게 보이지만 동일한 표기
- 동일한 의미를 가지지만, 하나의 줄로 표현할 수 있는 코드를 두 개의 줄로 표현한 경우

```python
result = Dense(128)(input)

==

dense = Dense(128)
result = dense(input)
```



