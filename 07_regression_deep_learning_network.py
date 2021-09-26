import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 예시데이터: 2018년 지역별 인구증가율(X)과 고령인구비율(Y)
X = [0.3, -0.78, 1.26, 0.03, 1.11, 0.24, -0.24, -0.47, -0.77, -0.37,
                  -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
Y = [12.27, 14.44, 11.87, 18.75, 17.52, 16.37, 19.78, 19.51, 12.65, 14.74,
                  10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=6, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(units=1)
])
# 1번째 레이어: 뉴런 6개, 활성화 함수 tanh
# 2번째 레이어: 뉴런 1개(X입력값에 대한 하나의 y값만 출력해야 하므로)

# tanh이란 하이퍼볼릭 탄젠트 함수. -1~1 사이의 출력을 반환하며, 시그모이드 함수와 유사한 형태를 갖는다.
# tanh(x) = (e^x - e^(-x))/(e^x+e^(-x))

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')
# 모델을 동작시킴. 활성화함수 -> SGD, 손실(error와 유사) -> MSE
# SGD -> Stochastic Gradient Descent -> 확률적 경사 하강법
# MSE -> Mean Squared Error -> 기대출력에서 실제 출력 뺀 뒤에 제곱한 값을 평균하는 것.
# 즉, error = y-ouput와 유사한 기능을 함.

model.summary()
model.fit(X, Y, epochs=10)
print(model.predict(X))

# 그래프 그리기
line_x = np.arange(min(X), max(X), 0.01)
line_y = model.predict(line_x)

plt.plot(line_x, line_y, 'r-')
plt.plot(X,Y,'bo')
plt.xlabel('Population Growth Rate')
plt.ylabel('Elderly Population Rate')
plt.show()