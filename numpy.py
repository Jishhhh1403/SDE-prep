import numpy as np

arr=np.array([1,2,3,4,5])

a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

print(b[1:4])

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5:2])

arr = np.array(['apple', 'banana', 'cherry'])

print(arr.dtype)