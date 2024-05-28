import numpy as np

mat = np.array([[1, 2, 3], [4, 5, 6]])
print(mat)

# 첫번째 행 출력
slicing_mat = mat[0, :]
print(slicing_mat)

# 두번째 열 출력
slicing_mat = mat[:, 1]
print(slicing_mat)