from keras import Sequential
from keras import layers
import numpy as np
import random
import matplotlib.pyplot as plt


x_train = np.loadtxt("Dataset/input.csv", delimiter = ',')
y_train = np.loadtxt("Dataset/labels.csv", delimiter = ',')

x_test = np.loadtxt("Dataset/input_test.csv", delimiter = ',')
y_test = np.loadtxt("Dataset/labels_test.csv", delimiter = ',')

# print("Shape of x_train: ", x_train.shape)
# print("Shape of y_train: ", y_train.shape)
# print("Shape of x_test: ", x_test.shape)
# print("Shape of y_test: ", y_test.shape)

x_train =x_train.reshape(len(x_train), 100, 100 , 3)
y_train =y_train.reshape(len(y_train), 1)

x_test =x_test.reshape(len(x_test), 100, 100 , 3)
y_test = y_test.reshape(len(y_test), 1)

x_train = x_train/255.0
x_test = x_test/255.0

print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of x_test: ", x_test.shape)
print("Shape of y_test: ", y_test.shape)

# idx = random.randint(0,len(x_train))
# plt.imshow(x_train[idx,:])
# plt.show()

model = Sequential([
    layers.Conv2D(32,(3,3), activation='relu', input_shape= (100,100,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(32,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')
     
    ])
    
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 10, batch_size= 64)

model.evaluate(x_test, y_test)