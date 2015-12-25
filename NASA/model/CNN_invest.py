# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from my_core_layer.speed_activation import *
from my_core_layer.PLRelu import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import  h5py
import os

import cPickle

data_augmentation = True
batch_size = 32
nb_classes = 4
nb_epoch = 50
img_channels = 3

# input image dimensions
img_rows, img_cols = 28, 28
"""
pkl_file = open('/home/dell/exs/NASA/data_prepare/SAT-6_all_RGB.pkl', 'rb')

# the data, shuffled and split between tran and test sets
X_train = cPickle.load(pkl_file)
Y_train = cPickle.load(pkl_file)
X_test = cPickle.load(pkl_file)
Y_test = cPickle.load(pkl_file)
"""
file_name = '/home/dell/exs/NASA/data_prepare/SAT-4_all_RGB.hdf5'
f = h5py.File(file_name,'r')
X_train = f['X_train'][:]
Y_train = f['y_train'][:]
X_test = f['X_test'][:]
Y_test = f['y_test'][:]

f.close()

print(X_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(Y_train.shape)

# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
y_test = np_utils.categorical_probas_to_classes(Y_test)
print(y_test)
#building CNN model
print("sat4-lrelu")
print('Building model......')
model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(img_channels, img_rows, img_cols)))
model.add(Lrelu())
model.add(Convolution2D(32, 3, 3, input_shape=(img_channels, img_rows, img_cols)))
model.add(Lrelu())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Convolution2D(64, 3, 3))
model.add(Lrelu())
model.add(Convolution2D(64, 3, 3))
model.add(Lrelu())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())

model.add(Dense(512))
model.add(Lrelu())
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))


# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
'''
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
'''

if not data_augmentation:
    print("Not using data augmentation or normalization")

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1,validation_data=(X_test,Y_test))

else:
    print("Using real time data augmentation")
    testAccu = []
    trainAccu = []
    testLoss = []
    trainLoss = []
    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        train_flag = 0
        trainAcAll = 0
        trainLoAll = 0
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.flow(X_train, Y_train,batch_size = batch_size):
            train_flag += 1
            trainLo , trainAc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            trainAcAll += trainAc
            trainLoAll += trainLo
            progbar.add(X_batch.shape[0], values=[("train loss", trainLo), ("train accuracy:", trainAc)])

        print("Testing...")
        # test time!
        testLo,testAc = model.evaluate(X_test,Y_test,show_accuracy=True)
        print ('test loss: ' + str(testLo) + '    ' + 'test Accuracy: ' + str(testAc))
        testAccu.append(testAc)
        trainAccu.append(trainAcAll/train_flag)

        testLoss.append(testLo)
        trainLoss.append(trainLoAll/train_flag)

    Pred = model.predict_classes(X_test)

# draw acc loss figure
    p_testAcc, = plt.plot(testAccu, color='blue')
    p_trainAcc, = plt.plot(trainAccu, color='red')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.legend([p_testAcc, p_trainAcc], ('testAcc', 'trainAcc'), 'best', numpoints=1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('/home/dell/exs/NASA/log_records/SAT-4_2_RGB_Accu.eps', dpi=128)
#    plt.show()
    plt.figure()
    p_testLoss, = plt.plot(testLoss, color='blue')
    p_trainLoss, = plt.plot(trainLoss, color='red')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend([p_testLoss, p_trainLoss], ('testLoss', 'trainLoss'), 'best', numpoints=1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('/home/dell/exs/NASA/log_records/SAT-4_2_RGB_Loss.eps', dpi=128)
    print("figure saved")
# save weights
    model.save_weights('/home/dell/exs/NASA/log_records/SAT-4_2_RGB.hdf5',overwrite=True)
    print("weights saved")


# save class_name.txt and labels.txt to draw confusion matrix


    labelFile = '/home/dell/exs/NASA/log_records/labels_SAT-4_2.txt'
    f1 = open(labelFile,'w')
    print (Pred)
    print ("Pred length: " + str(len(Pred)))

    for i in range(len(Pred)):
        line = str(y_test[i]) + ' ' + str(Pred[i]) + '\n'
        f1.write(line)
    f1.close()
'''
    class_name = 'class_name.txt'
    f2 = open(class_name, 'w')
    names = []
    for img in os.listdir('/media/dell/cb552bf1-c649-4cca-8aca-3c24afca817b/dell/data/UCMerced_LandUse/res'):# need to be changed
        names.append(img)
    print ("names Length:" + str(len(names)))
    for i in range(len(names)):
        line = str(names[i]) +  '\n'
        f2.write(line)
    f2.close()
'''

'''
    plt.plot(testAccu,color = 'blue')
    plt.plot(trainAccu,color = 'red')
    plt.savefig('/home/dell/exs/NASA/log_records/6classRGBAccu.png', dpi=128)
    plt.figure()
    plt.plot(testLoss,color = 'blue')
    plt.plot(trainLoss,color = 'red')
    plt.savefig('/home/dell/exs/NASA/log_records/6classRGBLoss.png', dpi=128)
    model.save_weights('/home/dell/exs/NASA/log_records/6classRGB.hdf5',overwrite=True)
'''