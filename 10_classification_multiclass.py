import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# UCI의 와인 데이터 세트를 사용.
# 외부에서 데이터를 불러오고 정제하는 과정.
red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
                  sep=';')
white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
                    sep=';')
# print(red.head())
# print(white.head())

# 와인이 레드(0)와인인지 화이트(1)와인인지 표시하는 속성('type')을 추가한다.
red['type'] = 0
white['type'] = 1
# print(red.head(2))
# print(white.head(2))

# 두 데이터프레임을 합친 새로운 데이터프레임 wine을 새롭게 정의한다.
wine = pd.concat([red, white])
# print(wine.describe())

# 다항 분류를 위해 '품질' 항목을 시각화
# print(wine['quality'].describe())
# print(wine['quality'].value_counts())
# plt.hist(wine['quality'], bins=7, rwidth=0.8)
# plt.show()

# 시각화 결과, 데이터의 양이 적고 범주가의 수가 너무 많은데다 각 데이터의 숫자가 차이가 난다.
# 따라서, 세 가지 범주로 다시 나누어 분류를 하도록 하자.
# 품질 3~5는 나쁨, 6은 보통, 7~9는 좋음으로 바꾼다.

wine.loc[wine['quality'] <= 5, 'new_quality'] = 0
wine.loc[wine['quality'] == 6, 'new_quality'] = 1
wine.loc[wine['quality'] >= 7, 'new_quality'] = 2
# 데이터프레임에 사용되는 .loc은 특정한 데이터의 인덱스를 골라내는 역할을 한다.
# 대괄호 안에 인수를 하나만 넣으면 행을 골라내고, 쉼표를 포함한 두 개의 인수를 넣으몇 차례대로 행,열을 골라낸다.

# print(wine['new_quality'].describe())
# print(wine['new_quality'].value_counts())

# 아래는 loc 예시.
# data = [['Apple', 11], ['Banana', 23], ['Coconut', 35]]
# df = pd.DataFrame(data, columns = ['Fruit', 'Count'])
# print(df)
# print()
# print(df.loc[0])
# print()
# print(df.loc[0, 'Fruit'])

del wine['quality']
wine_norm = (wine-wine.min()) / (wine.max() - wine.min())
wine_shuffle = wine_norm.sample(frac=1)
wine_np = wine_shuffle.to_numpy()

train_idx = int(len(wine_np)*0.8)
train_X, train_Y = wine_np[:train_idx, :-1], wine_np[:train_idx, -1]
test_X, test_Y = wine_np[train_idx:, :-1], wine_np[train_idx:, -1]
train_Y = tf.keras.utils.to_categorical(train_Y, num_classes=3)
test_Y = tf.keras.utils.to_categorical(test_Y, num_classes=3)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=48, activation='relu', input_shape=(12,)),
    tf.keras.layers.Dense(units=24, activation='relu'),
    tf.keras.layers.Dense(units=12, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_X, train_Y, epochs=25, batch_size=32, validation_split=0.25)


plt.figure(figsize=(12,4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g--', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()

print(model.evaluate(test_X, test_Y))
