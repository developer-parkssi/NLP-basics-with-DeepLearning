import numpy as np

mat = np.array([[1, 2], [4, 5], [7, 8]])
print(mat)

# 1행 0열의 원소
# => 0부터 카운트하므로 두번째 행 첫번째 열의 원소.
print(mat[1, 0])

# mat[[2행, 1행],[0열, 1열]]
# 각 행과 열의 쌍을 매칭하면 2행 0열, 1행 1열의 두 개의 원소.
indexing_mat = mat[[2, 1],[0, 1]]
print(indexing_mat)