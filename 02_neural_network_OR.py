import tensorflow as tf
import numpy as np
import math


# 한 개의 뉴런(퍼셉트론)을 사용해 AND 연산을 구현하는 신경망 네트워크.


def sigmoid(x):
    # 시그모이드의 결과값(y) 의 범위는 늘 0에서 1 사이이다. (x의 값과 무관)
    # 따라서 성공을 1, 실패로 0으로 두는 binary한 결과를 표현하는데 용이하다.

    # 다만 최근에는 ReLU 함수가 더 많이 쓰이는 모양.
    return 1 / (1 + math.exp(-x))


x = np.array( [ [1,1], [1,0], [0,1], [0,0] ] )
y = np.array( [ [1], [1], [1], [0] ] )
w = tf.random.normal([2],0,1)
b = tf.random.normal([1],0,1)
b_x = 1

for i in range(2000):
    error_sum = 0
    for j in range(4):
        output = sigmoid(np.sum(x[j]*w)+b_x*b)
        error = y[j][0] - output
        w = w + x[j] * 0.1 * error
        b = b + b_x * 0.1 * error
        error_sum += error

    if i % 200 == 199:
        print(i, error_sum)

for i in  range(4):
    print("X: {} Y: {} Output: {}".format( x[i] , y[i], sigmoid(np.sum(x[i]*w)+b) ))
