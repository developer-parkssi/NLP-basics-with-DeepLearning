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
