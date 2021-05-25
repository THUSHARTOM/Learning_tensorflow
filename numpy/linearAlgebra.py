import numpy as np

# matrix multiplication

# a = np.ones((2,3))
# b = np.full((3,2), 2)
# c = np.matmul(a,b)
# print(c)

# Statistics
#
# stats = np.array([[1,2,3],[4,5,6]])
# print(np.min(stats))
#
# print(np.max(stats, axis=1))
# print(np.sum(stats, axis=0))

# Reshaping -As long as they have the same number of components
# you can change the dimensions

# before = np.array([[1,2,3,4],[5,6,7,8]])
# print(before)
# after = before.reshape(4,-1)
# print(after)

## Vertical stacking

# v1 = np.array([1,2,3,4])
# v2 = np.array([5,6,7,8])
#
# print(np.vstack([v1,v2,v1,v2]))

## Horizontal stack

# h1 = np.ones((2,4))
# h2 = np.zeros((2,2))
#
# hstack = np.hstack((h1,h2))
# print(hstack)

## Index with list or True or False

a = np.array([1,2,3,4,5,6,7,8,9])
print(a[[0,1,2]])