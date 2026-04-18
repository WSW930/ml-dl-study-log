### 신경망 모델 훈련

## 손실 곡선

# 패션 MNIST 데이터 호출 및 훈련-테스트 세트 추출 & 전처리
import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2)




## 심층 신경망 객체 생성 함수 정의

def model_fn(a_layer = None):
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))
  model.add(keras.layers.Dense(100, activation = 'relu'))
  if a_layer:
    model.add(a_layer)
  model.add(keras.layers.Dense(10, activation = 'softmax'))
  return model

model = model_fn()
model.summary()
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])



## 훈련 손실 관찰 - History 클래스 객체를 활용

# fit() 메서드 => History 클래스 객체 반환
history = model.fit(train_scaled, train_target, epochs= 5, verbose = 0)

print("히스토리 Dictionary: key 카테고리: ", history.history.keys())

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Train loss per epochs')
plt.show()

plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Train accuracy per epochs')
plt.show()

# 에포크 횟수를 20회로 늘릴 때 훈련 손실 관찰

model = model_fn()
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(train_scaled, train_target, epochs= 20, verbose = 0)

plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Train loss per epochs(epochs = 20)')
plt.show()


## 검증 손실 관찰 - 과대적합 추이 파악

model = model_fn()
model.summary()
model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(train_scaled, train_target, epochs= 20, verbose = 0, validation_data=(val_scaled, val_target))

print("히스토리 Dictionary: key 카테고리(검증 세트 추가): ", history.history.keys())

plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('train loss VS val loss')
plt.legend()
plt.show()

# 옵티마이저 변경 후 검증 손실 재차 관찰

model = model_fn()
model.summary()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(train_scaled, train_target, epochs= 20, verbose = 0, validation_data=(val_scaled, val_target))

plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('train loss VS val loss(Optimizer: Adam)')
plt.legend()
plt.show()



## 드롭아웃 - 랜덤한 뉴런 OFF로 과대적합 방지

model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(train_scaled, train_target, epochs= 20, verbose = 0, validation_data=(val_scaled, val_target))

plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('train loss VS val loss(Optimizer: Adam) (Dropout-applyed)') # 드롭아웃: 과대적합 억제 효과
plt.legend()
plt.show()



## 모델 저장과 복원

model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(train_scaled, train_target, epochs= 10, verbose = 0, validation_data=(val_scaled, val_target))

model.save('model-whole.keras') # 모델 구조 + 파라미터를 모두 저장하는 save 메서드
model.save_weights('model-weights.weights.h5') # 모델 파라미터만 저장하는 save_weights

!ls -al model* # 세이브 파일 확인

# Case 1 - 새로운 신경망 모델 객체에 '모델 파라미터'만 로드

model = model_fn(keras.layers.Dropout(0.3)) # 신경망 모델 객체 초기화(새로 생성)
model.load_weights('model-weights.weights.h5')

import numpy as np

val_labels = np.argmax(model.predict(val_scaled), axis = -1)
# argmax 함수를 활용하여 각 샘플의 최대 확률 predict 인덱스 반환

print("검증 정확도: ", np.mean(val_labels == val_target))

# Case 2 - 새로운 신경망 모델 객체에 '모델 구조' 전체를 로드

model = keras.models.load_model('model-whole.keras')
print("검증 정확도2: ", model.evaluate(val_scaled, val_target))



## 콜백(Callback)

# ModelCheckpoint 콜백 예제
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# ModelCheckpoint 함수 => 매 에포크마다
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras', save_best_only = True)

model.fit(train_scaled, train_target, epochs= 20, verbose = 0, validation_data=(val_scaled, val_target), callbacks = [checkpoint_cb])

model = keras.models.load_model('best-model.keras')

print("ModelCheckpoint 콜백 기반 객체 검증 정확도", model.evaluate(val_scaled, val_target))



# ModelCheckpoint + EarlyStopping(조기종료) 콜백 예제
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.keras', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True)

history = model.fit(train_scaled, train_target, epochs= 20, verbose = 0, validation_data=(val_scaled, val_target), callbacks = [checkpoint_cb, early_stopping_cb])

print(early_stopping_cb.stopped_epoch)

plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('train loss VS val loss(Callback-applyed)')
plt.legend()
plt.show()