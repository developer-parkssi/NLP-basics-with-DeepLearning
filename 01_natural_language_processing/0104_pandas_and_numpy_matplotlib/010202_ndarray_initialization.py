import numpy as np

# 모든 값이 0인 2x3 배열 생성.
zero_mat = np.zeros((2,3))
print(zero_mat)

# 모든 값이 1인 2x3 배열 생성.
one_mat = np.ones((2,3))
print(one_mat)

# 모든 값이 특정 상수인 배열 생성. 이 경우 7.
same_value_mat = np.full((2,2), 7)
print(same_value_mat)

# 대각선 값이 1이고 나머지 값이 0인 2차원 배열을 생성.
eye_mat = np.eye(3)
print(eye_mat)

# 임의의 값으로 채워진 배열 생성
random_mat = np.random.random((2,2)) # 임의의 값으로 채워진 배열 생성
print(random_mat)

# 임의의 값으로 채워진 배열 생성
random_mat = np.random.random((2,2)) # 임의의 값으로 채워진 배열 생성
print(random_mat)