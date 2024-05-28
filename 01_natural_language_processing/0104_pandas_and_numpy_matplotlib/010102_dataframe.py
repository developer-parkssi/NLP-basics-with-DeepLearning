import pandas as pd

values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(values, index=index, columns=columns)

print('데이터프레임 출력 :')
print('-'*18)
print(df)

print('데이터프레임의 인덱스 : {}'.format(df.index))
print('데이터프레임의 열이름: {}'.format(df.columns))
print('데이터프레임의 값 :')
print('-'*18)
print(df.values)