import numpy as np
import serial
import time
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.utils import to_categorical

'''
This is  a Recurrent Neural Network (RNN). They are dessignd to work with
sequential data, where the order of data points matters. They have loops that
allow information to persist, making them suitable for tasks like natural language
processing, speech recognition, and time series predition. This is good, because
IMU data is inherently sequential in nature.
CNN is good if you have more than 1 IMU

Maybe I can do a hybrid, where I have a FNN for the EMG sensors and an RNN for the IMU
'''


column_limit=11

#calculate the number of sensors (IMU readings)
num_sensors = column_limit -2

time_steps = 1

def load_data_with_column_limit(file_path, column_limit):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            if len(row) == column_limit:
                data.append(row)
    return np.array(data, dtype=float)

def reshapedata(data, time_steps, num_imu_readings):
    num_samples = len(data) // time_steps  # Calculate the number of samples based on time_steps
    print("line 46: " , len(data))
    print(data.shape)
    num_features = num_imu_readings  # Number of features: EMG value + 9 IMU readings
    reshaped_data = data[:num_samples * time_steps, 0:num_features].reshape(num_samples, time_steps, num_features)
    return np.array(reshaped_data, dtype=float)


#Getting traning data
noClasses = int(input('Please enter the number of classes:-'));
X_train = np.array([])
y_train = np.array([])
for noClass in range (0,noClasses):
    if(noClass == 0):
        fileID = 'HO0.csv';
    elif(noClass == 1):
        fileID = 'HC0.csv';
    elif(noClass == 2):
        fileID = 'up0.csv';
    elif(noClass == 3):
        fileID = 'down0.csv';
    elif(noClass == 4):
        fileID = 'left0.csv';
    elif(noClass == 5):
        fileID = 'right0.csv';
    print(fileID);
    #data = np.loadtxt(fileID, delimiter=',', skiprows=1, dtype=float)
    data = load_data_with_column_limit(fileID, column_limit)
    print(data.shape , "\n")
    #data=reshapedata(data)
    #print("This is the reshaped data: ", data)
    l=len(data);


    temptarget = np.full((l,), noClass, dtype='int')

    if X_train.size == 0:
        X_train = data;
        y_train = temptarget;
    else:
        X_train = np.concatenate((data, X_train), axis=0)
        y_train = np.concatenate((temptarget, y_train), axis=0)

print("xtrain: ", X_train.shape)
print("ytrain:   ", y_train.shape)

# One-hot encode the target labels
y_train = to_categorical(y_train, num_classes=int(noClasses))


#print("this is y_train after being one-hot encoded:\n",y_train)

#getting testing data
X_test = np.array([])
y_test = np.array([])

for noClass in range (0,noClasses):
    if(noClass == 0):
        fileID = 'HO2.csv';
    elif(noClass == 1):
        fileID = 'HC2.csv';
    elif(noClass == 2):
        fileID = 'up2.csv';
    elif(noClass == 3):
        fileID = 'down2.csv';
    elif(noClass == 4):
        fileID = 'left2.csv';
    elif(noClass == 5):
        fileID = 'right2.csv';
    print(fileID);
    #data = np.loadtxt(fileID, delimiter=',', skiprows=1, dtype=float)
    data = load_data_with_column_limit(fileID, column_limit)
    print(data.shape , "\n")
    l=len(data);
    temptarget = np.full((l,), noClass, dtype='int')

    if X_test.size == 0:
        X_test = data;
        y_test = temptarget;
    else:
        X_test = np.concatenate((data, X_test), axis=0)
        y_test = np.concatenate((temptarget, y_test), axis=0)


print("xtest: ", X_test.shape)
print("ytest:   ", y_test.shape)
#print("temptarget:   ", temptarget)

# One-hot encode the target labels
y_test = to_categorical(y_test, num_classes=int(noClasses))


#reshape training and testing data
X_train = reshapedata(X_train, time_steps, num_imu_readings=9)
X_test = reshapedata(X_test, time_steps, num_imu_readings=9)

print("xtrain reshaped: ", X_train.shape)
print("xtest reshaped:   ", X_test.shape)


#Build the model
'''model = Sequential()
model.add(SimpleRNN(units=64, input_shape=(time_steps, features)))
model.add(Dense(units=num_classes, activation='softmax'))'''
model = Sequential()
model.add(SimpleRNN(units=64, activation='tanh', input_shape=(time_steps,num_sensors,)))  # Here, we use 'num_sensors' as input_shape
model.add(Dense(7, activation='sigmoid', input_shape=(num_sensors,)))
model.add(Dense(units=int(noClasses), activation='softmax'))
'''
model.add(Dense(int(noClasses), activation='softmax'))
'''

model.summary()

#Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])



#print("x test: ", X_test)
#print("y test: ", y_test)
# Train the model
model.fit(X_train, y_train, epochs=200, verbose=1, validation_data=(X_test, y_test))
accuracy = model.evaluate(X_train, y_train)


#history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
#accuracy = model.evaluate(X_train, y_train, verbose=0) #verbose=0 is to turn off the output progress bars during training
#accuracy = model.evaluate(X_train, y_train)

print('Accuracy: %.2f' % (accuracy[1] * 100))

#Make predictions
fileID="right3.csv"
X=load_data_with_column_limit(fileID, column_limit)
#X = np.loadtxt(fileID, delimiter=',', skiprows=1, dtype=float)
#X = reshapedata(X, time_steps)
X=reshapedata(X, time_steps, num_imu_readings=9)
#l=len(X);
#X=X[int(l/2):l];
predictions = model.predict(X)
print("predictions: \n")
print(predictions)
predicted_classes = np.argmax(predictions, axis=1)
print("predicted classes: \n",predicted_classes)
Prob = np.zeros(len(predictions), dtype='int')
#print("Actual classes:", Prob) this isn't the actual class
for i in range(1,len(predictions)):
        out = predictions[i,:];
        print(max(out))
        if max(out) > 0.7:
            a=np.array([max(out) == out]).astype(int)
            b=[range(1,int(noClasses)+1)]
            out = np.sum(a*b);
        else:
            out = Prob[1];
        length = Prob.size;
        Prob = np.concatenate(([out] ,Prob[0:length-1]), axis=0)
