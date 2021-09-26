import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 예시데이터: 2018년 지역별 인구증가율(X)과 고령인구비율(Y)
X = [0.3, -0.78, 1.26, 0.03, 1.11, 0.24, -0.24, -0.47, -0.77, -0.37,
                  -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
Y = [12.27, 14.44, 11.87, 18.75, 17.52, 16.37, 19.78, 19.51, 12.65, 14.74,
                  10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

# X, Y의 평균을 구합니다.
x_bar = sum(X) / len(X)
y_bar = sum(Y) / len(Y)

"""# 최소제곱법으로 a,b를 구하는 방법
a = sum([ (y-y_bar) * (x-x_bar) for y,x in list(zip(Y, X)) ])
a /= sum([ (x-x_bar) ** 2 for x in X ])
b = y_bar - a*x_bar
print('a:',a,'b:',b)
"""

# a와 b를 랜덤한 값으로 초기화합니다.
a = tf.Variable(random.random())
b = tf.Variable(random.random())
c = tf.Variable(random.random())

# 잔차의 제곱의 평균을 반환하는 함수입니다.
def compute_loss():
    y_pred = a * X * X + b * X + c
    loss = tf.reduce_mean((Y - y_pred) ** 2)
    return loss


optimizer = tf.optimizers.Adam(lr=0.07)
for i in range(1000):
    # 잔차의 제곱의 평균을 최소화(minimize)합니다.
    optimizer.minimize(compute_loss, var_list=[a,b,c])

    if i % 100 == 99:
        print(i, 'a:', a.numpy(), 'b:', b.numpy(), 'loss:', compute_loss().numpy())


# 그래프를 그리기 위해 회귀선의 x,y 데이터를 구합니다.
line_x = np.arange(min(X), max(X), 0.01)
line_y = a * line_x * line_x + b * line_x + c


# 붉은 색 실선으로 회귀선을 그립니다.
plt.plot(line_x, line_y, 'r-')

# 점 그래프를 그립니다.
plt.plot(X, Y, 'bo')
plt.xlabel('Population Growth Rate')
plt.ylabel('Elderly Population Rate')
plt.show()
