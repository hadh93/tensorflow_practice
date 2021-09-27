import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# UCI의 와인 데이터 세트를 사용.
# 외부에서 데이터를 불러오고 정제하는 과정.
red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
#print(red.head())
#print(white.head())

# 와인이 레드(0)와인인지 화이트(1)와인인지 표시하는 속성('type')을 추가한다.
red['type'] = 0
white['type'] = 1
# print(red.head(2))
# print(white.head(2))

# 두 데이터프레임을 합친 새로운 데이터프레임 wine을 새롭게 정의한다.
wine = pd.concat([red, white])
# print(wine.describe())

# 데이터프레임 wine의 type 속성만 히스토그램으로 출력한다.
# 현재 데이터셋에서는 레드보다 화이트와인이 약 3배 많은 상황.
plt.hist(wine['type'])
plt.xticks([0,1])
plt.show()
# print(wine['type'].value_counts())

# 정규화 실행 전, 모든 값이 숫자인지를 확인한다.
# 숫자가 아닌 값이 있으면 정규화 과정에 에러가 발생한다.
# 현재 모두 non-null이고, float64거나 int64이므로 정규화 진행 가능하다.
# print(wine.info())

wine_norm = ( wine - wine.min() ) / ( wine.max() - wine.min() )
# print(wine_norm.head())
# print(wine_norm.describe()) # 속성들의 min값이 0, max값이 1이므로 정규화가 잘 진행되었다.

# pandas의 sample() 함수는 전체 데이터프레임에서 frac 인수로 지정된 비율만큼의 행을 랜덤하게 뽑아 새로운 데이터 프레임을 만든다.
# frac = 1 로 설정되어 있기 때문에, 100%의 데이터가 다 사용되어 섞인다.
# 즉, 모든 데이터를 뽑아 섞는 역할을 수행한다.
wine_shuffle = wine_norm.sample(frac=1)
# print(wine_shuffle.head())

# numpy의 to_numpy()함수는 데이터프레임을 넘파이 array로 변환하여 반환한다.
wine_np = wine_shuffle.to_numpy()
# print(wine_np[:5])

# 데이터 셋의 80%를 학습 데이터로, 20%를 테스트 데이터로 분할하는 과정.
train_idx = int(len(wine_np)*0.8) # 데이터셋을 나눌 구분점이 되는 인덱스. 데이터셋의 80% 지점.
train_X, train_Y = wine_np[:train_idx, :-1], wine_np[:train_idx, -1]
# 구분점까지의 모든 IV들 (마지막 한칸을 제외한 전부)은 X에, 구분점까지의 모든 DV들 (마지막 한 칸)은 Y에 저장.
test_X, test_Y = wine_np[train_idx:, :-1], wine_np[train_idx:, -1]
# 구분점 이후의 모든 IV들 (마지막 한칸을 제외한 전부)은 X에, 구분점 이후의 모든 DV들 (마지막 한 칸)은 Y에 저장.
# print(train_X[0])
# print(train_Y[0])
# print(test_X[0])
# print(test_Y[0])

# 정답 행렬 (Y값 행렬)을 '원-핫 인코딩(One-Hot Encoding)' 방식으로 바꾼다.
# 원 핫 인코딩이란: 정답에 해당하는 인덱스의 값에는 1을, 나머지 인덱스에는 모두 0을 넣는 방식이다.
# 즉, [1. 0.] 은 0번이 정답이므로 레드와인. 반대로 [0. 1.] 은 1번이 정답이므로 화이트 와인이다.
train_Y = tf.keras.utils.to_categorical(train_Y, num_classes=2)
test_Y = tf.keras.utils.to_categorical(test_Y, num_classes=2)
# print(train_Y[0])
# print(test_Y[0])

# 딥러닝 학습 시작.
# 시퀸스 모델 사용.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=48, activation='relu', input_shape=(12,)),
    tf.keras.layers.Dense(units=24, activation='relu'),
    tf.keras.layers.Dense(units=12, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='softmax')
])
# 분류모델이므로 마지막 레이어의 활성화 함수로 소프트맥스(softmax)를 사용한다.
# 소프트맥스 함수: 출력값들을 자연로그의 밑인 e의 지수로 사용해 계산한 뒤 모둔 더한 값으로 나눈다.
# 간단히 말하자면 -> 큰 값을 강화하고, 작은 값은 약화한다.

# 원-핫 인코딩을 사용하므로 마지막 뉴런의 개수는 1개가 아니라 2개이다.
# 예측률 예시:
# 정답-[1, 0] 분류 네트워크 예측-[1,0] -> 100% 예측률
# 정답-[1, 0] 분류 네트워크 예측-[0,1] -> 0% 예측률


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='categorical_crossentropy', metrics=['accuracy'])

# 크로스 엔트로피에 관한 내용은 다시 학습해두자.
# 정보이론에서 엔트로피란: 불확실성을 숫자로 정량화한 것.
#                      확률의 역수에 로그를 취한 값.
#                      확률의 역수를 취하는 이유는, 확률이 높을 수록 정보량(놀라움)이 적다고 판단하기 때문.
#                        예: 비가 올 확률이 1%일 때, 비가 오지 않을 확률은 99%.
#                            이 경우 각 사건의 정보량은
#                            비가 내림: -log(0.01) = 4.605, 비가 오지 않음: -log(0.99) = 0.010
#                            즉, 비가 내리는 것은 460배 더 놀라운 사건이 됨.

# 분류문제는 정확도(accuracy)가 곧 퍼포먼스이므로, 이를 설정하는 것은 필수이다.

model.summary()

history = model.fit(train_X, train_Y, epochs=25, batch_size=32, validation_split=0.25)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], 'g--', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()