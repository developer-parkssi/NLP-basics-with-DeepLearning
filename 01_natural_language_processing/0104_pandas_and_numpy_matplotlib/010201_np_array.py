# 1차원 배열
import numpy as np

vec = np.array([1, 2, 3, 4, 5])
print(vec)

# 2차원 배열
mat = np.array([[10, 20, 30], [ 60, 70, 80]])
print(mat)

print('vec의 타입 :', type(vec))
print('mat의 타입 :', type(mat))

print('vec의 축의 개수 :',vec.ndim) # 축의 개수 출력
print('vec의 크기(shape) :',vec.shape) # 크기 출력

print('mat의 축의 개수 :',mat.ndim) # 축의 개수 출력
print('mat의 크기(shape) :',mat.shape) # 크기 출력