import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

DATA_PATH="https://raw.githubusercontent.com/urtbest86/MLOps/master/result_train_dataset2.csv"
df = pd.read_csv(DATA_PATH)

DATA_PATH="https://raw.githubusercontent.com/urtbest86/MLOps/master/result_test_dataset2.csv"
test = pd.read_csv(DATA_PATH)

train=df.sample(frac=0.8)
val=df.sample(frac=0.2)

mean = train.mean(axis=0)
train -= mean
std = train.std(axis=0)
train /= std

test -= mean
test /= std

val-=mean
val/=std

train_data_set = train.values
x_train = train_data_set[:, 2:-1].astype(float)
y_train = train_data_set[:, -1].astype(float)

test_data_set = test.values
x_test = test_data_set[:, 2:-1].astype(float)
y_test = test_data_set[:, -1].astype(float)

val_data_set = val.values
x_val = val_data_set[:, 2:-1].astype(float)
y_val = val_data_set[:, -1].astype(float)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1024, input_dim = 4, activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam',metrics=['mae'])

eph_size=150
hist=model.fit(x_train, y_train, epochs=eph_size, batch_size=4,validation_data=(x_val,y_val))



