import numpy as np
from datetime import datetime
from itertools import count

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 2], [3, 4], [5, 6]])
c = np.array([[7], [8], [9]])
d = np.array([10, 11, 12])

print(a)
print(b)
print(c)
print(d)

e = [[1, 2, 3], [4, 5, 6]]
print(e)
e1 = np.stack(e, axis=0)
print("e1=", e1)
e2 = np.stack(e, axis=1)
print("e2=", e2)

print("ab")
print(np.dot(a, b))

print("ac")
print(np.dot(a, c))

print("ad")
print(np.dot(a, d))

print("np.pi * a")
print(np.pi * a)

state = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

coeff = np.loadtxt("tests/coeff.txt", delimiter=" ")
print("coeff.shape=", coeff.shape)

print("np.dot(coeff, state)=", np.dot(coeff, state))

print("np.cos(np.dot(coeff, state))=", np.cos(np.dot(coeff, state)))

print("np.cos(np.dot(np.pi * coeff, state))=", np.cos(np.dot(np.pi * coeff, state)))

experiment_time = str(datetime.now()).split(".")[0] + "-" + str(datetime.now()).split(".")[1]

print("experiment_time=", experiment_time)

iid = count()

print("iid=", next(iid))

print("iid=", next(iid))
