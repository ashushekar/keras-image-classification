import os
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten

BATCHSIZE = 32
NUMCLASSES = 10
EPOCHS = 100
NUM_PREDICTIONS = 20
SAVEDIR = os.path.join(os.getcwd(), 'saved_models')
MODELNAME = 'keras_cifar10_trained_model.h5'

# Split the data between train and test set
(trainX, trainY), (testX, testY) = cifar10.load_data()
print("trainX shape: {}".format(trainX.shape))
print("trainY shape: {}".format(trainY.shape))
print("testX shape: {}".format(testX.shape))
print("testY shape: {}".format(testY.shape))

# Convert labels to one hot encode
trainY = keras.utils.to_categorical(trainY, NUMCLASSES)
testY = keras.utils.to_categorical(testY, NUMCLASSES)

# Model structure
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=trainX.shape[1:],
                 padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=trainX.shape[1:],
                 padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=NUMCLASSES, activation='softmax'))

# initiate RMSpropOptimizer
optimizer = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Normalise data
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX = trainX / 255
testX = testX / 255

model.fit(trainX, trainY, batch_size=BATCHSIZE, epochs=EPOCHS,
          validation_data=(testX, testY), shuffle=True)

# Save model and weights
if not os.path.isdir(SAVEDIR):
    os.makedirs(SAVEDIR)
MODELPATH = os.path.join(SAVEDIR, MODELNAME)
model.save(MODELPATH)

scores = model.evaluate(testX, testY, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



