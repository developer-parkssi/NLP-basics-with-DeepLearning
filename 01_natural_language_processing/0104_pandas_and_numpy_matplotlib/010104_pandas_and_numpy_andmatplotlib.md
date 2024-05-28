# Pandas
## 1) 시리즈(Series)
- 시리즈 클래스는 1차원 배열의 값(values)에 각 값에 대응되는 인덱스(index)를 부여할 수 있는 구조
```angular2html
sr = pd.Series([17000, 18000, 1000, 5000],
               index=["피자", "치킨", "콜라", "맥주"])
print('시리즈 출력 :')
print('-'*15)
print(sr)

[출력]
시리즈 출력 :
---------------
피자    17000
치킨    18000
콜라     1000
맥주     5000
dtype: int64
```
## 2) 데이터프레임(DataFrame)
- 데이터프레임은 2차원 리스트를 매개변수로 전달
- 행방향 인덱스(index)와 열방향 인덱스(column)가 존재
```
values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(values, index=index, columns=columns)

print('데이터프레임 출력 :')
print('-'*18)
print(df)

[출력]
데이터프레임 출력 :
------------------
       A  B  C
one    1  2  3
two    4  5  6
three  7  8  9
```
- 생성된 데이터프레임으로부터 인덱스(index), 값(values), 열(columns)을 각각 출력
```
print('데이터프레임의 인덱스 : {}'.format(df.index))
print('데이터프레임의 열이름: {}'.format(df.columns))
print('데이터프레임의 값 :')
print('-'*18)
print(df.values)

[출력]
데이터프레임의 인덱스 : Index(['one', 'two', 'three'], dtype='object')
데이터프레임의 열이름: Index(['A', 'B', 'C'], dtype='object')
데이터프레임의 값 :
------------------
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```
## 3) 데이터프레임의 생성
- 리스트(List), 시리즈(Series), 딕셔너리(dict), Numpy의 ndarrays, 또 다른 데이터프레임으로부터 생성
- 이중 리스트로 생성하는 경우
```
# 리스트로 생성하기
data = [
    ['1000', 'Steve', 90.72], 
    ['1001', 'James', 78.09], 
    ['1002', 'Doyeon', 98.43], 
    ['1003', 'Jane', 64.19], 
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14],
]

df = pd.DataFrame(data)
print(df)

[출력]
      0         1      2
0  1000     Steve  90.72
1  1001     James  78.09
2  1002    Doyeon  98.43
3  1003      Jane  64.19
4  1004  Pilwoong  81.30
5  1005      Tony  99.14
```
- 생성된 데이터프레임에 열(columns)을 지정 가능
```
df = pd.DataFrame(data, columns=['학번', '이름', '점수'])
print(df)

[출력]
     학번        이름     점수
0  1000     Steve  90.72
1  1001     James  78.09
2  1002    Doyeon  98.43
3  1003      Jane  64.19
4  1004  Pilwoong  81.30
5  1005      Tony  99.14
```
- 딕셔너리를 통해 데이터프레임 생성
```
# 딕셔너리로 생성하기
data = {
    '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
    '이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
    '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]
    }

df = pd.DataFrame(data)
print(df)

[출력]
     학번        이름     점수
0  1000     Steve  90.72
1  1001     James  78.09
2  1002    Doyeon  98.43
3  1003      Jane  64.19
4  1004  Pilwoong  81.30
5  1005      Tony  99.14
```
## 5) 외부 데이터 읽기
- Pandas는 CSV, 텍스트, Excel, SQL, HTML, JSON 등 다양한 데이터 파일을 읽고 데이터 프레임을 생성 가능
```
df = pd.read_csv('example.csv')
print(df)

[출력]
   student id      name  score
0        1000     Steve  90.72
1        1001     James  78.09
2        1002    Doyeon  98.43
3        1003      Jane  64.19
4        1004  Pilwoong  81.30
5        1005      Tony  99.14
```
- 인덱스 자동 부여
```
print(df.index)

[출력]
RangeIndex(start=0, stop=6, step=1)
```
# 2. 넘파이(Numpy)
- 넘파이(Numpy)는 수치 데이터를 다루는 파이썬 패키지
Numpy의 핵심이라고 불리는 다차원 행렬 자료구조인 ndarray를 통해 벡터 및 행렬을 사용하는 선형 대수 계산에서 주로 사용
## 1) np.array()
- np.array()는 리스트, 튜플, 배열로 부터 ndarray를 생성
```
# 1차원 배열
vec = np.array([1, 2, 3, 4, 5])
print(vec)

[출력]
[1 2 3 4 5]
```
- 2차원 배열
```
# 2차원 배열
mat = np.array([[10, 20, 30], [ 60, 70, 80]]) 
print(mat)

[출력]
[[10 20 30]
 [60 70 80]]
```
- 배열 타입
```
print('vec의 타입 :',type(vec))
print('mat의 타입 :',type(mat))

vec의 타입 : <class 'numpy.ndarray'>
mat의 타입 : <class 'numpy.ndarray'>
```
- Numpy 배열에는 축의 개수(ndim)와 크기(shape)라는 개념이 존재
- 배열의 크기를 정확히 숙지하는 것은 딥 러닝에서 매우 중요
```
print('vec의 축의 개수 :',vec.ndim) # 축의 개수 출력
print('vec의 크기(shape) :',vec.shape) # 크기 출력

[출력]
vec의 축의 개수 : 1
vec의 크기(shape) : (5,)
```
```
print('mat의 축의 개수 :',mat.ndim) # 축의 개수 출력
print('mat의 크기(shape) :',mat.shape) # 크기 출력

[출력]
mat의 축의 개수 : 2
mat의 크기(shape) : (2, 3)
```
## 2) ndarray의 초기화
- ndarray를 만드는 다양한 다른 방법이 존재
- np.zeros()는 배열의 모든 원소에 0을 삽입
```
# 모든 값이 0인 2x3 배열 생성.
zero_mat = np.zeros((2,3))
print(zero_mat)

[출력]
[[0. 0. 0.]
 [0. 0. 0.]]
```
- np.ones()는 배열의 모든 원소에 1을 삽입
```
# 모든 값이 1인 2x3 배열 생성.
one_mat = np.ones((2,3))
print(one_mat)

[출력]
[[1. 1. 1.]
 [1. 1. 1.]]
```
- np.full()은 배열에 사용자가 지정한 값을 삽입
```
# 모든 값이 특정 상수인 배열 생성. 이 경우 7.
same_value_mat = np.full((2,2), 7)
print(same_value_mat)

[출력]
[[7 7]
 [7 7]]
```
- np.eye()는 대각선으로는 1이고 나머지는 0인 2차원 배열
```
# 대각선 값이 1이고 나머지 값이 0인 2차원 배열을 생성.
eye_mat = np.eye(3)
print(eye_mat)

[출력]
[[1. 0. 0.]
 [0. 1. 0.]]
 [0. 0. 1.]]
```
- np.random.random()은 임의의 값을 가지는 배열을 생성
```
# 임의의 값으로 채워진 배열 생성
random_mat = np.random.random((2,2)) # 임의의 값으로 채워진 배열 생성
print(random_mat)

[출력]
[[0.3111881  0.72996102]
 [0.65667734 0.40758328]]
```
## 3) np.arange()
- np.arange(n)은 0부터 n-1까지의 값을 가지는 배열을 생성
```
# 0부터 9까지
range_vec = np.arange(10)
print(range_vec)

[출력]
[0 1 2 3 4 5 6 7 8 9]
```
- np.arange(i, j, k)는 i부터 j-1까지 k씩 증가하는 배열을 생성
```
# 1부터 9까지 +2씩 적용되는 범위
n = 2
range_n_step_vec = np.arange(1, 10, n)
print(range_n_step_vec)

[출력]
[1 3 5 7 9]
```
## 4) np.reshape()
- np.reshape()은 내부 데이터는 변경하지 않으면서 배열의 구조 바꿈
```
# 0부터 29까지의 숫자를 생성하는 arange(30)을 수행한 후, 원소의 개수가 30개이므로 5행 6열의 행렬로 변경
reshape_mat = np.array(np.arange(30)).reshape((5,6))
print(reshape_mat)

[출력]
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]]
```
## 5) Numpy 슬라이싱
- ndarray를 통해 만든 다차원 배열은 파이썬의 자료구조인 리스트처럼 슬라이싱(slicing) 기능을 지원
```
mat = np.array([[1, 2, 3], [4, 5, 6]])
print(mat)

[출력]
[[1 2 3]
 [4 5 6]]
```
```
# 첫번째 행 출력
slicing_mat = mat[0, :]
print(slicing_mat)

[출력]
[1 2 3]
```
```
# 두번째 열 출력
slicing_mat = mat[:, 1]
print(slicing_mat)

[출력]
[2 5]
```
## 6) Numpy 정수 인덱싱(integer indexing)
- 연속적이지 않은 원소로 배열을 만들 경우에는 슬라이싱으로는 생성 x
- 이런 경우에는 인덱싱을 사용하여 배열을 구성 가능
```
mat = np.array([[1, 2], [4, 5], [7, 8]])
print(mat)

[출력]
[[1 2]
 [4 5]
 [7 8]]
```
- 특정 위치의 원소
```
# 1행 0열의 원소
# => 0부터 카운트하므로 두번째 행 첫번째 열의 원소.
print(mat[1, 0])

[출력]
4
```
- 특정 위치의 원소 두 개를 가져와 새로운 배열
```
# mat[[2행, 1행],[0열, 1열]]
# 각 행과 열의 쌍을 매칭하면 2행 0열, 1행 1열의 두 개의 원소.
indexing_mat = mat[[2, 1],[0, 1]]
print(indexing_mat)

[출력]
[7 5]
```
## 7) Numpy 연산
- Numpy를 사용하면 배열간 연산을 손쉽게 수행 가능
- np.add(), np.subtract(), np.multiply(), np.divide() 사용 가능
```
x = np.array([1,2,3])
y = np.array([4,5,6])

# result = np.add(x, y)와 동일.
result = x + y
print(result)

[출력]
[5 7 9]
```
```
# result = np.subtract(x, y)와 동일.
result = x - y
print(result)

[출력]
[-3 -3 -3]
```
```
# result = np.multiply(result, x)와 동일.
result = result * x
print(result)

[출력]
[-3 -6 -9]
```
```
# result = np.divide(result, x)와 동일.
result = result / x
print(result)

[출력]
[-3. -3. -3.]
```
- *를 통해 수행한 것은 요소별 곱
- Numpy에서 벡터와 행렬곱 또는 행렬곱을 위해서는 dot() 사용
```
mat1 = np.array([[1,2],[3,4]])
mat2 = np.array([[5,6],[7,8]])
mat3 = np.dot(mat1, mat2)
print(mat3)

[출력]
[[19 22]
 [43 50]]
```