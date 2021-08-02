import matplotlib.pyplot as plt
import tensorflow as tf
import time

x = range(20)
# x의 범위: 0~19
y = tf.random.normal([20], 0, 1)
# y는 길이 20짜리 1차원 벡터. 평균 0, 표준편차 1 범위의 random draw.

# plt.plot은 기본적으로 꺾은선 그래프이다.
plt.plot(x, y)
plt.show()
time.sleep(3)

# 점 그래프로도 변환 가능. 'blue o'라는 뜻
plt.plot(x, y, 'bo') # ro로 바꾸면 빨간 점, yo로 바꾸면 노란 점...
plt.show()

import numpy as np
import tensorflow as tf
import math

x = np.array( [ [1,1], [1,0], [0,1], [0,0] ] )
y = np.array( [ [0], [1], [1], [0] ] )

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape= (2,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mse')
# 모델을 동작시킴. 활성화함수 -> SGD, 손실(error와 유사) -> MSE
# SGD -> Stochastic Gradient Descent -> 확률적 경사 하강법
# MSE -> Mean Squared Error -> 기대출력에서 실제 출력 뺀 뒤에 제곱한 값을 평균하는 것.
# 즉, error = y-ouput와 유사한 기능을 함.

model.summary()
# 모델이 어떻게 생겼는지 확인. 에러가 표시된다면 코드에 문제가 잇는 것임.

history = model.fit(x, y, epochs=2000, batch_size=1)
# 학습을 진행시키는 코드. 총 2000회 진행.

print(model.predict(x))
# 학습 결과 평가

for weight in model.weights:
    print(weight)

plt.plot(history.history['loss'])
plt.show()