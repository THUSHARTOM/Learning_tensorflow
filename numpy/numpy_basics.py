import numpy as np

a = np.ones((5,5))
z = np.zeros((3,3))
a[1:4,1:4]=z
a[2,2] = 9
b = np.full_like(a,5)
c = np.random.rand(4,2)
d = np.random.randint(2,5, size=(3,3))
print(a)







