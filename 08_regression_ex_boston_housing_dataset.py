from tensorflow.keras.datasets import boston_housing
(train_X, train_Y), (test_X, test_Y) = boston_housing.load_data()
import tensorflow as tf
import matplotlib.pyplot as plt

# (train_X, train_Y)는 훈련 데이터이다. 이는 학습 과정에 사용되는 데이터라는 뜻.
# (test_X, test_Y)는 테스트 데이터이다. 이는 학습 결과를 평가하기 위한 데이터다.

# 훈련/테스트 데이터 이외에도 검증(validation)데이터가 존재한다.
# 이는 훈련 데이터 셋의 일부로, 학습이 잘 되는지 검증하는 역할로 사용된다.
# 검증 데이터의 성적이 잘 나오지 않는다면 학습을 중지하는 것도 가능하다.

print(len(train_X), len(test_X))
print(train_X[0])
print(train_Y[0])

# 현재는 훈련데이터 404개, 테스트데이터 102개로 약 8:2의 비율이다.
# 훈련 데이터의 일부를 떼어서 검증 데이터를 만들어야 한다.
# 보통 이 비율은 훈련/검증/테스트 데이터의 비융르 60/20/20 으로 한다.


# 데이터는 그 단위가 각각 다르므로, 데이터를 전처리하여 정규화(standardization)해야 효율이 좋다.
# 각 데이터에서 평균값을 뺀 뒤 표준편차로 나눈다.
# 이하는 현 데이터 세트의 정규화 전처리 코드.

x_mean = train_X.mean(axis=0)
x_std = train_X.std(axis=0)
train_X -= x_mean
train_X /= x_std
test_X -= x_mean
test_X /= x_std

y_mean = train_Y.mean(axis=0)
y_std = train_Y.std(axis=0)
train_Y -= y_mean
train_Y /= y_std
test_Y -= y_mean
test_Y /= y_std

print(train_X[0])
print(train_Y[0])


# 딥러닝 학습 준비를 위해 모델 구축

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=52, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dense(units=39, activation='relu'),
    tf.keras.layers.Dense(units=26, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
# 레이어 뉴런 수: 52개, 39개, 26개, 1개 순
# 활성화 함수로는 ReLU가 사용되고 있다.
# ReLU란 정류 선형 유닛 (Rectified Linear Unit)에 대한 함수.
# ReLU는 입력값이 0보다 작으면 0으로 출력, 0보다 크면 입력값 그대로 출력하는 함수(즉, y=x).

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')

model.summary()

# 학습 시작
history = model.fit(train_X,train_Y, epochs=25, batch_size=32, validation_split=0.25)
# validation_split이란 훈련 데이터의 25%정도를 검증 데이터로 떼서 학습결과를 검증한다는 뜻.

callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')]
# 콜백(callback)함수를 사용하면 학습 도중의 개입을 할 수 있다.
# 콜백함수는 에포크가 끝날 때마다 호출된다.
# EarlyStopping 함수는 학습을 일찍 멈추는 기능을 하는데,
# monitor에 들어간 속성이 개선될 기회를 patience에 들어간 값의 횟수만큼 준다.
# 즉, 여기서는 val_loss가 3차례의 epoch에서 개선되지 않는다면 멈춘다.

plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

model.evaluate(test_X, test_Y)

pred_Y = model.predict(test_X)
plt.figure(figsize=(5,5))
plt.plot(test_Y, pred_Y, 'b.')
plt.axis([min(test_Y), max(test_Y), min(test_Y), max(test_Y)])

# y = x에 해당하는 대각선
plt.plot([min(test_Y), max(test_Y)], [min(test_Y), max(test_Y)], ls="--", c=".3")
plt.xlabel('test_Y')
plt.ylabel('pred_Y')

plt.show()