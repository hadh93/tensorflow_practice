# 시작하기에 앞서:
# XOR 문제는 '인공지능의 겨울'을 불러온 것으로 잘 알려져 있다.
# 그 핵심은, 하나의 퍼셉트론은 XOR 연산자도 만들어낼 수 없다는 점이다.
# (Marvin Minsky & Seymour Papert 의 'Perceptron'이라는 책에 증명된 바 있음)

# 해결책은 여러 개의 퍼셉트론을 사용하는 것이다! 3 개의 퍼셉트론과 뉴런을 사용하여 XOR 문제를 해결해보자.


import numpy as np
import tensorflow as tf
import math

x = np.array( [ [1,1], [1,0], [0,1], [0,0] ] )
y = np.array( [ [0], [1], [1], [0] ] )

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape= (2,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')
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