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


rgs=model.compile(loss='mse', optimizer='adam',metrics=['mae'])

eph_size=150
hist=model.fit(x_train, y_train, epochs=eph_size, batch_size=4,validation_data=(x_val,y_val))


laoss_nd_metrics = model.evaluate(x_test, y_test, batch_size=32)



from azureml.core.run import Run
# start an Azure ML run
run = Run.get_context()

# log a single value
run.log("Final test loss(mse)", laoss_nd_metrics[0])
print('Test loss(mse):', laoss_nd_metrics[0])

run.log('Final test loss(mae)', laoss_nd_metrics[1])
print('Test loss(mae):', laoss_nd_metrics[1])


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 3))
plt.title('Mosquito with Keras MLP ({} epochs)'.format(eph_size), fontsize=14)
plt.plot(hist.history['loss'], 'b-', label='loss', lw=4, alpha=0.5)
plt.plot(hist.history['val_loss'], 'r--', label='val_loss', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
run.log_image('loss VS val_loss', plot=plt)


import joblib

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=rgs, filename='outputs/Youjin_test_model.pkl')

'''import matplotlib.pyplot as plt

plt.plot(model.predict(x_test),label='predict value')
plt.plot(y_test, label='real value')
plt.plot(model.predict(x_test))
plt.legend()
plt.show()


from azureml.core import Run

run = Run.get_context()

score = model.evaluate(x_test, y_test, verbose=0)

run.log_image('Accuracy vs Loss', plot=plt)'''
